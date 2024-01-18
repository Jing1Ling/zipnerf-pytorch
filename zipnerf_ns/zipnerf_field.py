import pickle
from typing import Dict, Literal, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from gridencoder.grid import GridEncoder
from internal.coord import pos_enc, track_linearize
# from internal.models import set_kwargs
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.field_components.field_heads import FieldHeadNames
import torch.nn.functional as F
from internal import ref_utils,coord
from internal import image
from internal import ref_utils
import torch.nn.init as init
try:
    from torch_scatter import segment_coo
except:
    pass
def set_kwargs(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
class ZipNerfField(nn.Module):
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 256  # The width of the second part of MLP.
    skip_layer_dir: int = 0  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = True  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    scale_featurization: bool = False
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21
    net_width_glo: int = 128  # The width of the second part of MLP.
    net_depth_glo: int = 2  # The width of the second part of MLP.
    rand: bool = False
    # from nerfstudio-field_component-encodings.py
    implementation: Literal["tcnn", "torch"] = "tcnn"
    
            
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()
        set_kwargs(self, kwargs)  # if there was input some params, will be set automaticlly

        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:
            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)
            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]

        self.grid_num_levels = int(
            np.log(self.grid_disired_resolution / self.grid_base_resolution) / np.log(self.grid_level_interval)) + 1
        self.encoder = GridEncoder(input_dim=3,
                                   num_levels=self.grid_num_levels,
                                   level_dim=self.grid_level_dim,
                                   base_resolution=self.grid_base_resolution,
                                   desired_resolution=self.grid_disired_resolution,
                                   log2_hashmap_size=self.grid_log2_hashmap_size,
                                   gridtype='hash',
                                   align_corners=False)

        last_dim = self.encoder.output_dim
        # if self.scale_featurization:
        #     last_dim += self.encoder.num_levels
        self.density_layer = nn.Sequential(nn.Linear(last_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64,
                                                     1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
        
        last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)

        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)

            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)

            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)

            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0

            last_dim_rgb += dim_dir_enc

            if self.use_n_dot_v:
                last_dim_rgb += 1

            if self.num_glo_features > 0:
                last_dim_glo = self.num_glo_features
                for i in range(self.net_depth_glo - 1):
                    self.register_module(f"lin_glo_{i}", nn.Linear(last_dim_glo, self.net_width_glo))
                    last_dim_glo = self.net_width_glo
                self.register_module(f"lin_glo_{self.net_depth_glo - 1}",
                                     nn.Linear(last_dim_glo, self.bottleneck_width * 2))

            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, means, stds, rand=False, no_warp=False):
        """Helper function to output density."""
        # Encode input positions
        if self.warp_fn is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound

        features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
        weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * self.encoder.grid_sizes ** 2))
        
        features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        # if self.scale_featurization:
        #     with torch.no_grad():
        #         vl2mean = segment_coo((self.encoder.embeddings ** 2).sum(-1),
        #                               self.encoder.idx,
        #                               torch.zeros(self.grid_num_levels, device=weights.device),
        #                               self.grid_num_levels,
        #                               reduce='mean'
        #                               )
        #     featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoder.init_std ** 2 + vl2mean).sqrt()
        #     features = torch.cat([features, featurized_w], dim=-1)
        x = self.density_layer(features)
            
        raw_density = x[..., 0]  # Hardcoded to a single channel.
        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x, means.mean(dim=-2)

    def forward(self,
                rand,
                means, stds,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None,
                no_warp=False):        
        """
        Args:
        rand: if random .
        means: [..., n, 3], coordinate means.
        stds: [..., n], coordinate stds.
        viewdirs: [..., 3], if not None, this variable will
            be part of the input to the second part of the MLP concatenated with the
            output vector of the first part of the MLP. If None, only the first part
            of the MLP will be used with input x. In the original paper, this
            variable is the view direction.
        imageplane:[batch, 2], xy image plane coordinates
            for each ray in the batch. Useful for image plane operations such as a
            learned vignette mapping.
        glo_vec: [..., num_glo_features], The GLO vector for each ray.
        exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.
        """
        
        if self.disable_density_normals:
            raw_density, x, means_contract = self.predict_density(means, stds, rand=rand, no_warp=no_warp)
            raw_grad_density = None
            normals = None
        else:
            with torch.enable_grad():
                means.requires_grad_(True)
                raw_density, x, means_contract = self.predict_density(means, stds)
                d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
                raw_grad_density = torch.autograd.grad(
                    outputs=raw_density,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
            raw_grad_density = raw_grad_density.mean(-2)
            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -ref_utils.l2_normalize(raw_grad_density)

        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            self.normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            self.normals_to_use = normals

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)
        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = (F.softplus(raw_roughness + self.roughness_bias))

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if self.rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    # Append GLO vector if used.
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f"lin_glo_{i}")(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                    bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift

                    x = [bottleneck]
                else:
                    x = []
                # # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], self.normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)
                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        self.normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x.append(dotprod)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)
            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding 
            
        res=dict(
            coord=means_contract,
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )
        return res
