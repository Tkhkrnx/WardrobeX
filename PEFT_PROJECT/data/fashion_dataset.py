import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FashionDataset(Dataset):
    def __init__(self, extractor=None, root='outputs/train/fashiongen_crops'):
        """
        :param split: 'train' or 'val'，用来切换不同子目录
        :param extractor: transformers 的 ViTFeatureExtractor，用于标准预处理
        :param root: 根目录，默认是 detect_modanet_and_crop.py 的输出路径
        """
        self.extractor = extractor
        # 根据你的说明，root 就是最终目录，不需要再拼接 split
        self.image_dir = root
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"找不到图像目录: {self.image_dir}")

        self.image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])
        if len(self.image_paths) == 0:
            raise RuntimeError(f"目录 {self.image_dir} 下没有裁剪图像")

        # fallback transform，如果没有 extractor
        self.default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.extractor:
            pixel_values = self.extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            pixel_values = self.default_transform(image)

        return {
            "pixel_values": pixel_values,
            "image_name": os.path.basename(image_path)
        }
