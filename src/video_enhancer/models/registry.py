"""Model registry to instantiate target networks dynamically."""

import torch.nn as nn
from video_enhancer.config.schema import TrainingConfig, TaskType

# We will import specific models as we build them in Phase 4-7.
# For Phase 3 testing, we use a dummy model.


class DummyModel(nn.Module):
    """A trivial model to test the training loop."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)


from video_enhancer.models.pretrained.realesrgan import build_realesrgan
from video_enhancer.models.pretrained.nafnet import build_nafnet

def create_model(config: TrainingConfig) -> nn.Module:
    """Instantiate the requested model architecture."""
    ckpt_path = str(config.checkpoint.pretrained_path) if config.checkpoint.pretrained_path else None
    
    if config.task == TaskType.SUPER_RESOLUTION:
        return build_realesrgan(
            checkpoint_path=ckpt_path,
            freeze_config=config.freeze,
            lora_config=config.lora,
            sequence_length=getattr(config.data, "sequence_length", 1),
            use_gradient_checkpointing=getattr(config, "gradient_checkpointing", False)
        )
    elif config.task == TaskType.DEBLUR:
        return build_nafnet(
            checkpoint_path=ckpt_path,
            freeze_config=config.freeze,
            lora_config=config.lora,
            use_gradient_checkpointing=True
        )
    elif config.task == TaskType.FACE_ENHANCEMENT:
        from video_enhancer.models.pretrained.gfpgan import build_gfpgan
        return build_gfpgan(
            checkpoint_path=ckpt_path,
            freeze_config=config.freeze,
            lora_config=config.lora
        )
    elif config.task == TaskType.INTERPOLATION:
        from video_enhancer.models.pretrained.rife import build_rife
        return build_rife(
            checkpoint_path=ckpt_path,
            freeze_config=config.freeze,
            lora_config=config.lora
        )
    else:
        raise NotImplementedError(f"Model for task {config.task.value} not yet implemented.")
