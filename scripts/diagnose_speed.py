import time
from pathlib import Path
from video_enhancer.config.loader import load_config
from video_enhancer.data.dataset import VideoPatchDataset
from torch.utils.data import DataLoader

config = load_config(Path("configs/finetune_realesrgan.yaml"))
dataset = VideoPatchDataset(
    data_dir=config.data.train_dir,
    data_config=config.data,
    deg_config=config.degradation,
    scale=4
)

print(f"Dataset size: {len(dataset)}")

# Time single __getitem__ call directly (no multiprocessing)
t0 = time.time()
sample = dataset[0]
print(f"Single __getitem__: {time.time()-t0:.2f}s")

t0 = time.time()
sample = dataset[1]
print(f"Second __getitem__ (idx=1): {time.time()-t0:.2f}s")

# Time a full batch via DataLoader with your actual config
loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True, persistent_workers=True)
it = iter(loader)
t0 = time.time()
batch = next(it)
print(f"First batch (cold workers): {time.time()-t0:.2f}s")
t0 = time.time()
batch = next(it)
print(f"Second batch (warm workers): {time.time()-t0:.2f}s")
t0 = time.time()
batch = next(it)
print(f"Third batch (warm workers): {time.time()-t0:.2f}s")
