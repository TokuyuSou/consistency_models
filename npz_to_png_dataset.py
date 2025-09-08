import numpy as np, os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Convert NPZ to PNG dataset.")
parser.add_argument("npz_path", nargs="?", default="samples_10k_32x32x3.npz", help="Path to the .npz file")
parser.add_argument("out_dir", nargs="?", default="samples_mnist32rgb", help="Output directory for PNG images")
args = parser.parse_args()

npz_path = args.npz_path
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

arr = np.load(npz_path)["arr_0"]  

if arr.ndim==4 and arr.shape[1]==3 and arr.shape[2]==32:
    # NCHW -> NHWC
    arr = arr.transpose(0,2,3,1)
assert arr.ndim==4 and arr.shape[-1]==3 and arr.shape[1]==32 and arr.shape[2]==32

if arr.dtype!=np.uint8:
    arr = np.clip(arr,0,255).astype(np.uint8)

for i, img in enumerate(arr):
    Image.fromarray(img).save(os.path.join(out_dir, f"{i:06d}.png"))
print(f"wrote {len(arr)} images to {out_dir}")