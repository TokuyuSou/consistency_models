# save_mnist_as_imagefolder.py
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path

root = Path("/workspace/consistency_models/datasets/mnist32rgb")   # ここを好みの出力先に
for split in ["train", "test"]:
    ds = datasets.MNIST(
        root="./",
        train=(split=="train"),
        download=True,
        transform=transforms.ToTensor()
    )
    for i, (x, y) in enumerate(ds):
        # x: [1, 28, 28] -> PIL -> 32x32 -> RGB
        img = transforms.ToPILImage()(x)
        img = Image.new("L", (32, 32), 0)  # 0埋めで 28→32 (周辺パディング)
        img.paste(transforms.ToPILImage()(x), (2, 2))  # 中央寄せ
        img = img.convert("RGB")  # 3ch に複製

        out = root / split / str(y)
        out.mkdir(parents=True, exist_ok=True)
        img.save(out / f"{i:07d}.png")
