from dataclasses import dataclass, field
from typing import Dict, List, Literal, Type
from internal.render import cast_rays, compute_alpha_weights
from zipnerf_ns.zipnerf_field import  ZipnerfMLPWrapper

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type
from nerfacc import RaySamples, importance_sampling
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from internal import coord
from internal import coord
from internal import train_utils
from internal import render
from internal import stepfun
from internal.models import MLP, NerfMLP,PropMLP
import numpy as np
from torch.utils._pytree import tree_map
from nerfstudio.utils import colormaps

try:
    from torch_scatter import segment_coo
except:
    print('Failed to import torch_scatter')

def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


@dataclass
class ZipNerfModelConfig(ModelConfig):
    near_plane: float = 0.2 # Near plane distance.
    far_plane: float = 1000000.0 # Far plane distance.
    num_prop_samples: int = 64  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = 'power_transformation'  # The curve used for ray dists.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    # learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_max_num_iters = 2500
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    proposal_weights_anneal_max_num_iters = 1000
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    distinct_prop: bool = True  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = True  # If true, make the background opaque.
    power_lambda: float = -1.5
    std_scale: float = 0.5
    prop_desired_grid_size = [512, 2048]
    compute_extras = True
    
    # from config.py
    interlevel_loss_mult: float = 0.0  # Mult. for the loss on the proposal MLP.
    anti_interlevel_loss_mult: float = 0.01  # Mult. for the loss on the proposal MLP.
    pulse_width = [0.03, 0.003]  # Mult. for the loss on the proposal MLP.
    distortion_loss_mult: float = 0.005  # Multiplier on the distortion loss.
    orientation_loss_mult: float = 0.0  # Multiplier on the orientation loss.
    orientation_coarse_loss_mult: float = 0.0  # Coarser orientation loss weights.
    # What that loss is imposed on, options are 'normals' or 'normals_pred'.
    orientation_loss_target: str = 'normals_pred'
    hash_decay_mults: float = 0.1
    opacity_loss_mult: float = 0.  # Multiplier on the distortion loss.
    
    predicted_normal_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
    # Mult. on the coarser predicted normal loss.
    predicted_normal_coarse_loss_mult: float = 0.0
    disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
    
    data_loss_type: str = 'charb'  # What kind of loss to use ('mse' or 'charb').
    compute_disp_metrics: bool = False  # If True, load and compute disparity MSE.
    compute_normal_metrics: bool = False  # If True, load and compute normal MAE.
    charb_padding: float = 0.001  # The padding used for Charbonnier loss.
    data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
    data_coarse_loss_mult: float = 0.  # Multiplier for the coarser data terms.

    zero_glo: bool = True
    rand: bool = False
    gradient_scaling: bool = False
    importance_sampling: bool = False
    dpcpp_backend: bool = False
    vis_num_rays: int = 16  # The number of rays to visualize.
    _target: Type = field(default_factory=lambda: ZipNerfModel)


class ZipNerfModel(Model):
    """ZipNerf Model."""

    config: ZipNerfModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # initiate backend
        from extensions import Backend
        
        Backend.set_backend('dpcpp' if self.config.dpcpp_backend else 'cuda')
        self.backend = Backend.get_backend()

        # initiate MLP
        self.nerf_mlp = ZipnerfMLPWrapper(num_glo_features=self.config.num_glo_features,
                                num_glo_embeddings=self.config.num_glo_embeddings)
        
        if self.config.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif not self.config.distinct_prop: 
            self.prop_mlp = ZipnerfMLPWrapper()
        else:
            for i in range(self.config.num_levels - 1):
                self.register_module(f'prop_mlp_{i}', ZipnerfMLPWrapper(grid_disired_resolution=self.config.prop_desired_grid_size[i],
                                                                        disable_rgb=True,grid_level_dim=1))
        
        if self.config.num_glo_features > 0 and not self.config.zero_glo:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_vecs = nn.Embedding(self.config.num_glo_embeddings, self.config.num_glo_features)
 
        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        
        self.collider = NearFarCollider(near_plane=0.2, far_plane=1e6)
        self.step = 0

    def get_outputs(self, ray_bundle: RayBundle):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""

        ray_bundle.metadata["viewdirs"] = ray_bundle.directions
        ray_bundle.metadata["radii"] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        ray_bundle.directions = ray_bundle.directions * ray_bundle.metadata["directions_norm"]

        if self.training:
            self.config.rand = True

        device = ray_bundle.origins.device

        # set global vectors
        if self.config.num_glo_features > 0:
            if not self.config.zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = ray_bundle.camera_indices[..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(ray_bundle.origins.shape[:-1] + (self.config.num_glo_features,), device=device)
        else:
            glo_vec = None

        # Define the bijection from normalized to metric ray distance.
        t_to_s, s_to_t = coord.construct_ray_warps(self.config.raydist_fn, ray_bundle.nears, ray_bundle.fars, self.config.power_lambda)
        
        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        # # `near_anneal_rate` can be used to anneal in the near bound at the start
        # of training, eg. 0.1 anneals in the bound over the first 10% of training.
        if self.config.near_anneal_rate is None:
            init_s_near = 0.
        else:  
            init_s_near = np.clip(1 - np.clip(self.step/self.config.near_anneal_max_num_iters,0,1), 0, self.near_anneal_init)
            # init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0,
            #                       self.near_anneal_init)

        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(ray_bundle.nears, init_s_near),
            torch.full_like(ray_bundle.fars, init_s_far)
        ], dim=-1)
        weights = torch.ones_like(ray_bundle.nears)
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.config.num_levels):
            is_prop = i_level < (self.config.num_levels - 1)
            num_samples = self.config.num_prop_samples if is_prop else self.config.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.config.dilation_bias + self.config.dilation_multiplier * (
                    init_s_far - init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = self.config.dilation_bias > 0 or self.config.dilation_multiplier > 0
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # Optionally anneal the weights as a function of training iteration.
            if self.config.anneal_slope > 0:  
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                train_frac = np.clip(self.step*1.0/self.config.proposal_weights_anneal_max_num_iters, 0, 1)
                anneal = bias(train_frac, self.config.anneal_slope)
            else:
                anneal = 1.
            # anneal = 1.

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.config.resample_padding),
                torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            if self.config.importance_sampling:
                sdist = self.backend.funcs.sample_intervals(
                    self.config.rand,
                    sdist.contiguous(),
                    stepfun.integrate_weights(torch.softmax(logits_resample, dim=-1)).contiguous(),
                    num_samples,
                    self.single_jitter)
            else:
                sdist = stepfun.sample_intervals(
                    self.config.rand,
                    sdist,
                    logits_resample,
                    num_samples,
                    single_jitter=self.config.single_jitter,
                    domain=(init_s_near, init_s_far))
            
            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.config.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)
            # cone_radius = torch.sqrt(ray_bundle.pixel_area) * 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
            
            cone_radius = ray_bundle.metadata["radii"]
            # Cast our rays, by turning our distance intervals into Gaussians.
            cam_dirs = None
            means, stds, ts = render.cast_rays(
                tdist,
                ray_bundle.origins,
                ray_bundle.directions,
                cam_dirs,
                cone_radius,
                self.config.rand,
                std_scale=self.config.std_scale,
                rand_vec=None)
            # Push our Gaussians through one of our two MLPs.
            mlp = (self.get_submodule(
                f'prop_mlp_{i_level}') if self.config.distinct_prop else self.prop_mlp) if is_prop else self.nerf_mlp
            ray_results = mlp(
                self.config.rand,
                means, stds,
                viewdirs=ray_bundle.metadata["viewdirs"] if self.config.use_viewdirs else None,
                imageplane=None, # not used in mlp
                glo_vec=None if is_prop else glo_vec,
                exposure=None, # not used in mlp
            )
            
            if self.config.gradient_scaling:
                ray_results['rgb'], ray_results['density'] = train_utils.GradientScaler.apply(
                    ray_results['rgb'], ray_results['density'], ts.mean(dim=-1))
                # self.config.opaque_background = pickle.load(load_inforward)
            # Get the weights used by volumetric rendering (and our other losses).
            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                ray_bundle.directions,
                opaque_background=self.config.opaque_background,
            )[0]

            
            # Define or sample the background color for each ray.
            if self.config.bg_intensity_range[0] == self.config.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.config.bg_intensity_range[0]
            elif self.config.rand is None:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.config.bg_intensity_range[0] + self.config.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.config.bg_intensity_range[0]
                maxval = self.config.bg_intensity_range[1]
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval

            
            # Render each ray.
            rendering = render.volumetric_rendering(
                ray_results['rgb'],
                weights,
                tdist,
                bg_rgbs,
                ray_bundle.fars,
                self.config.compute_extras, 
                extras={
                    k: v
                    for k, v in ray_results.items()
                    if k.startswith('normals') or k in ['roughness']
                })
                   
            if self.config.compute_extras: 
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (
                    weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]
            
            if self.training:
                # Compute the hash decay loss for this level.
                idx = mlp.encoder.idx
                param = mlp.encoder.embeddings
                if self.config.dpcpp_backend:
                    ray_results['loss_hash_decay'] = (param ** 2).mean()
                else:
                    loss_hash_decay = segment_coo(param ** 2,
                                                    idx,
                                                    torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
                                                    reduce='mean'
                                                    ).mean()
                    ray_results['loss_hash_decay'] = loss_hash_decay
            renderings.append(rendering)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_history.append(ray_results)

        if self.config.compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]
        outputs={}
        outputs['rgb']=renderings[2]['rgb']
        outputs['depth']=renderings[2]['depth']
        outputs['acc']=renderings[2]['acc']
        outputs['distance_mean']=renderings[2]['distance_mean']
        outputs['distance_median']=renderings[2]['distance_median']
        outputs['renderings']=renderings
        outputs['ray_history'] = ray_history
        return outputs 
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        def set_step(step):
            self.step = step

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            )
        )
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
            """Returns the parameter groups needed to optimizer your model components."""
            param_groups = {}
            param_groups["model"] = list(self.parameters())
            return param_groups
    

    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""
        metrics_dict = {}
        gt_rgb = batch['image'].to(self.device)
        predicted_rgb = outputs['rgb']
        metrics_dict["psnr"] = self.psnr(gt_rgb, predicted_rgb)
        return metrics_dict
        
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""
        loss_dict={}
        batch['lossmult'] = torch.Tensor([1.]).to(self.device)
        
        data_loss, stats = train_utils.compute_data_loss(batch, outputs['renderings'], self.config)
        loss_dict['data'] = data_loss
        
        if self.training:
            # interlevel loss in MipNeRF360
            # if self.config.interlevel_loss_mult > 0 and not self.config.single_mlp:
            #     loss_dict['interlevel'] = train_utils.interlevel_loss(outputs['ray_history'], self.config)

            # interlevel loss in ZipNeRF360
            if self.config.anti_interlevel_loss_mult > 0 and not self.config.single_mlp:
                loss_dict['anti_interlevel'] = train_utils.anti_interlevel_loss(outputs['ray_history'], self.config)

            # distortion loss
            if self.config.distortion_loss_mult > 0:
                loss_dict['distortion'] = train_utils.distortion_loss(outputs['ray_history'], self.config)

            # opacity loss
            # if self.config.opacity_loss_mult > 0:
            #     loss_dict['opacity'] = train_utils.opacity_loss(outputs['rgb'], self.config)

            # # orientation loss in RefNeRF
            # if (self.config.orientation_coarse_loss_mult > 0 or
            #         self.config.orientation_loss_mult > 0):
            #     loss_dict['orientation'] = train_utils.orientation_loss(batch, self.config, outputs['ray_history'],
            #                                                             self.config)
            # hash grid l2 weight decay
            if self.config.hash_decay_mults > 0:
                loss_dict['hash_decay'] = train_utils.hash_decay_loss(outputs['ray_history'], self.config)

            # # normal supervision loss in RefNeRF
            # if (self.config.predicted_normal_coarse_loss_mult > 0 or
            #         self.config.predicted_normal_loss_mult > 0):
            #     loss_dict['predicted_normals'] = train_utils.predicted_normal_loss(
            #         self.config, outputs['ray_history'], self.config)
        return loss_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]: # type: ignore
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
        gt_rgb = batch["image"].to(self.device)
        # image = self.renderer_rgb.blend_background(image)

        predicted_rgb = outputs["rgb"]
        print('min,max:',predicted_rgb.min(),predicted_rgb.max())
        acc = colormaps.apply_colormap(outputs["acc"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["acc"])

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(gt_rgb, predicted_rgb).item()),
            "ssim": float(self.ssim(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0))),
            "lpips": float(self.lpips(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0)))
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict
