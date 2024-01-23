import pickle
from typing import Dict, Literal, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from gridencoder.grid import GridEncoder
from internal.coord import pos_enc, track_linearize
from internal.models import set_kwargs
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field,FieldConfig  # for custom Field
from nerfstudio.field_components.field_heads import FieldHeadNames
from dataclasses import dataclass
import torch.nn.functional as F
from internal import ref_utils,coord
from internal import image
from internal import ref_utils
from internal.models import MLP
import torch.nn.init as init
try:
    from torch_scatter import segment_coo
except:
    pass
# def set_kwargs(self, kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)

class ZipnerfMLPWrapper(MLP):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()
    #TODO: deal with parameter of prop & nerf mlp

