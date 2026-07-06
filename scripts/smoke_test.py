import os
import torch
import math
from pathlib import Path

from video_enhancer.config.loader import load_config
from video_enhancer.training.realesrgan_trainer import RealESRGANFineTuner

def dicts_are_close(d1, d2, rtol=1e-5, atol=1e-8):
    if type(d1) != type(d2): return False
    if isinstance(d1, dict):
        if set(d1.keys()) != set(d2.keys()): return False
        return all(dicts_are_close(d1[k], d2[k], rtol, atol) for k in d1)
    elif isinstance(d1, (list, tuple)):
        if len(d1) != len(d2): return False
        return all(dicts_are_close(v1, v2, rtol, atol) for v1, v2 in zip(d1, d2))
    elif isinstance(d1, torch.Tensor):
        return torch.allclose(d1.cpu(), d2.cpu(), rtol=rtol, atol=atol)
    elif isinstance(d1, (int, float, bool, str)):
        if isinstance(d1, float) and isinstance(d2, float) and math.isnan(d1) and math.isnan(d2): return True
        return d1 == d2
    else:
        return True # ignore other types

def test_smoke():
    config_path = Path("configs/finetune_realesrgan.yaml")
    config = load_config(config_path)
    
    # Force minimal setup
    config.data.batch_size = 1
    config.optimizer.gradient_accumulation_steps = 1
    config.data.num_workers = 0
    config.scheduler.total_steps = 10
    config.checkpoint.save_every_n_steps = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running smoke test on {device}")
    
    # 1. Setup Model, Optimizer, and Loader
    from video_enhancer.models.registry import create_model
    model = create_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    seq_len = getattr(config.data, "sequence_length", 1)
    num_ch = 3 * seq_len
    batch = {
        "lr": torch.randn(config.data.batch_size, num_ch, 64, 64),
        "hr": torch.randn(config.data.batch_size, 3, 256, 256)
    }
    
    class MockLoader:
        def __iter__(self): yield batch
        def __len__(self): return 1
    loader = MockLoader()
    
    tuner1 = RealESRGANFineTuner(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        val_loader=None
    )
    
    # 2. Run 1 step
    print("Running step 1...")
    config.scheduler.total_steps = 1
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    tuner1.fit()
    assert tuner1.global_step == 1, "Tuner did not complete 1 step"
    
    # 3. Save checkpoint
    print("Saving checkpoint...")
    ckpt_dir = Path("checkpoints/smoke_test")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Use tuner's built-in save
    tuner1.save_dir = ckpt_dir
    tuner1.save_checkpoint()
    ckpt_path = ckpt_dir / "step_000001.pth"
    
    # 4. Instantiate new tuner and resume
    print("Instantiating tuner2 and resuming...")
    model2 = create_model(config).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    tuner2 = RealESRGANFineTuner(
        config=config,
        model=model2,
        optimizer=optimizer2,
        train_loader=loader,
        val_loader=None
    )
    tuner2.load_checkpoint(ckpt_path)
    
    # 5. Assert identical weights
    print("Asserting identical states...")
    assert dicts_are_close(tuner1.model.state_dict(), tuner2.model.state_dict()), "model mismatch"
    assert dicts_are_close(tuner1.net_d.state_dict(), tuner2.net_d.state_dict()), "net_d mismatch"
    assert dicts_are_close(tuner1.optimizer.state_dict(), tuner2.optimizer.state_dict()), "optimizer_g mismatch"
    assert dicts_are_close(tuner1.opt_d.state_dict(), tuner2.opt_d.state_dict()), "optimizer_d mismatch"
    
    # 6. Run 1 more step
    print("Running step 2 on resumed tuner...")
    config.scheduler.total_steps = 2
    tuner2.fit()
    assert tuner2.global_step == 2, "Tuner did not complete step 2"
    
    # 7. Print VRAM
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak VRAM used during smoke test: {peak_mb:.2f} MB")
    
    # Cleanup
    ckpt_path.unlink()
    ckpt_dir.rmdir()
    print("Verification Passed: Smoke test successful!")

if __name__ == "__main__":
    test_smoke()
