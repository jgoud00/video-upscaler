import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
from video_enhancer.config.loader import load_config
from video_enhancer.models.registry import create_model

def test_onnx_export():
    config_path = Path("configs/finetune_realesrgan.yaml")
    config = load_config(config_path)
    
    # 1. Export model
    import subprocess
    onnx_path = Path("checkpoints/test_export.onnx")
    cmd = ["venv\\Scripts\\python.exe", "scripts\\export_onnx.py", "-c", str(config_path), "-o", str(onnx_path)]
    subprocess.run(cmd, check=True)
    
    # 2. Run PyTorch model
    device = torch.device("cpu")
    model = create_model(config).to(device)
    model.eval()
    
    seq_len = getattr(config.data, "sequence_length", 1)
    num_in_ch = 3 * seq_len
    dummy_input = torch.randn(1, num_in_ch, 256, 256)
    
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
        
    # 3. Run ONNX model
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    # 4. Compare
    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"Max absolute difference: {max_diff}")
    
    tolerance = 1e-4
    assert max_diff < tolerance, f"Difference too high: {max_diff}"
    print(f"Verification Passed: ONNX and PyTorch outputs match within tolerance {tolerance}.")
    
    onnx_path.unlink()

if __name__ == "__main__":
    test_onnx_export()
