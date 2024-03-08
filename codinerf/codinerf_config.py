from __future__ import annotations

from codinerf.codinerf_datamanager import (
    CoDiDataManagerConfig,
)

from codinerf.codinerf_model import CoDiModelConfig
from codinerf.codinerf_pipeline import (
    CoDiPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig, PatchPixelSamplerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig

codinerf = MethodSpecification(
    config=TrainerConfig(
        method_name="codinerf",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=17500,
        max_num_iterations=35001,
        steps_per_eval_all_images = 17500,
        mixed_precision=True,
        gradient_accumulation_steps = {'camera_opt':256,'proposal_networks':256}, #'proposal_networks':1024, 
        save_only_latest_checkpoint=False,
        pipeline=CoDiPipelineConfig(
            datamanager=CoDiDataManagerConfig(
                pixel_sampler=PatchPixelSamplerConfig(
                    num_rays_per_batch=4096,
                    ignore_mask = True,
                    fisheye_crop_radius = None #None kitti-705, parkinglot 910
                ),
                dataparser=NerfstudioDataParserConfig(
                    downscale_factor=4,
                    train_split_fraction=0.8,
                    load_3D_points=True,
                    depth_unit_scale_factor=1/256,
                    scene_scale = 2,
                    scale_factor = 2
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                masks_on_gpu=True,
                images_on_gpu=True,
                patch_size=32,
                
            ),
            model=CoDiModelConfig( 
                eval_num_rays_per_chunk=1 << 15,
                use_gradient_scaling=False,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15), 
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15), #1 << 15 1 << 12
        vis="wandb",
    ),
    description="CoDiNeRF method.",
)

