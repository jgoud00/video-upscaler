import sys
import argparse
import cv2
import os
import torch

# Monkey-patch basicsr compatibility BEFORE importing gfpgan
if "torchvision.transforms.functional_tensor" not in sys.modules:
    import torchvision.transforms.functional as functional
    sys.modules["torchvision.transforms.functional_tensor"] = functional

from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output image path")
    args = parser.parse_args()

    # 1. Run Background Upsampler (RealESRGAN) FIRST
    print("Initializing Real-ESRGAN Background Upsampler...")
    bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    bg_upsampler = RealESRGANer(
        scale=2,  
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=bg_model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

    # Read image
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"Could not read image: {args.input}")

    print(f"Step 1: Upscaling full image with Real-ESRGAN...")
    upscaled_img, _ = bg_upsampler.enhance(img, outscale=2)

    # 2. Setup GFPGAN for Face Enhancement (No Background Upsampler)
    print("Initializing GFPGAN Face Enhancer...")
    restorer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        upscale=1, # The image is already upscaled!
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    print(f"Step 2: Enhancing faces and pasting them back onto the upscaled image...")
    # Process
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        upscaled_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=1.0
    )

    # Save
    if restored_img is not None:
        cv2.imwrite(args.output, restored_img)
        print(f"Successfully saved full enhanced image to: {args.output}")
        
        if restored_faces and len(restored_faces) > 0:
            face_path = args.output.replace(".jpg", "_FaceOnly.jpg")
            cv2.imwrite(face_path, restored_faces[0])
            print(f"Saved the isolated face to: {face_path}")
    else:
        print("Failed to enhance image.")

if __name__ == '__main__':
    main()
