from torchvision import datasets
from PIL import Image
from pathlib import Path

# Output root directory (change as you like)
root = Path("/workspace/consistency_models/datasets/cifar10")

for split in ["train", "test"]:
    # CIFAR-10 provides 32x32 RGB images by default (when transform=None)
    ds = datasets.CIFAR10(
        root="./",
        train=(split == "train"),
        download=True,
        transform=None  # keep PIL Images as-is
    )

    for i, (img, y) in enumerate(ds):
        # Ensure image is a PIL Image in RGB mode
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")  # CIFAR-10 is already RGB, but this is a safe guard

        # Save to: {root}/{split}/{label_id}/{0000000}.png
        out = root / split / str(y)  # use numeric labels (0-9) as in the MNIST example
        out.mkdir(parents=True, exist_ok=True)
        img.save(out / f"{i:07d}.png")
