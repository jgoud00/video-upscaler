"""Unit tests for Phase 3 shared infrastructure (LoRA and BaseFineTuner)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from video_enhancer.config.schema import TrainingConfig, TaskType
from video_enhancer.models.adapters.lora import Conv2dLoRA, inject_lora
from video_enhancer.training.base_trainer import BaseFineTuner


class DummyDataset(Dataset):
    def __len__(self):
        return 10
        
    def __getitem__(self, idx):
        return {
            "hr": torch.rand(3, 16, 16),
            "lr": torch.rand(3, 8, 8)
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.body = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.tail = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.tail(self.body(self.conv1(x)))


class DummyTrainer(BaseFineTuner):
    def train_step(self, batch):
        out = self.model(batch["lr"])
        # Fake loss to ensure gradients flow
        loss = torch.nn.functional.mse_loss(out, torch.nn.functional.interpolate(batch["hr"], scale_factor=0.5))
        return {"mse": loss}


def test_lora_wrapper():
    # 1. Create a dummy model
    model = DummyModel()
    
    # 2. Inject LoRA into 'body' layers only
    model = inject_lora(model, target_module_names=["body"], rank=4)
    
    # 3. Verify types
    assert isinstance(model.conv1, nn.Conv2d)
    assert isinstance(model.body, Conv2dLoRA)
    assert isinstance(model.tail, nn.Conv2d)
    
    # 4. Verify parameter freezing
    # Base conv inside LoRA should be frozen
    assert next(model.body.base_conv.parameters()).requires_grad == False
    
    # A and B matrices should be trainable
    assert next(model.body.lora_A.parameters()).requires_grad == True
    assert next(model.body.lora_B.parameters()).requires_grad == True
    
    # 5. Forward pass
    x = torch.rand(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 3, 32, 32)


def test_base_trainer(tmp_path):
    # Setup config
    config = TrainingConfig(
        task=TaskType.SUPER_RESOLUTION,
        experiment_name="test_trainer",
        log_dir=tmp_path / "logs",
    )
    config.checkpoint.save_dir = tmp_path / "checkpoints"
    config.device.use_amp = False  # Keep false for CPU testing
    config.scheduler.total_steps = 5  # Just run 5 steps
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loader = DataLoader(DummyDataset(), batch_size=2)
    
    trainer = DummyTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader
    )
    
    trainer.fit()
    
    # Verify it ran 5 steps
    assert trainer.global_step == 5
