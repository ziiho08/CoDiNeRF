from dataclasses import dataclass, field
from typing import Type, Dict, List, Literal
from codinerf.codinerf_loss import clip_loss
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.data.scene_box import OrientedBox, SceneBox


@dataclass
class CoDiModelConfig(InstantNGPModelConfig): 
    use_clip_loss : bool = True
    near_plane: float = 0.05
    far_plane: float = 1000.0
    average_init_density: float = 0.01
    num_nerf_samples_per_ray: int = 128
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    _target: Type = field(default_factory=lambda: CoDiModel)

class CoDiModel(NGPModel):    

    config: CoDiModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        
        # losses
        self.clip_loss = clip_loss()
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
                
        if self.training:

            if self.config.use_clip_loss:
                loss_dict["clip_loss"] = self.clip_loss(gt_rgb, pred_rgb)

        return loss_dict