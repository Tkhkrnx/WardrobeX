import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
from peft import (
    get_peft_model,
    LoraConfig,
)
from data.fashion_dataset import FashionDataset
import numpy as np
from torchvision import transforms
from PIL import Image


class ContrastiveLoss(nn.Module):
    """对比损失函数"""

    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        计算对比损失 - 自监督方式
        features: [batch_size, feature_dim]
        """
        # 标准化特征
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建正样本对的掩码（假设相邻样本是正样本对）
        batch_size = features.shape[0]
        pos_mask = torch.zeros_like(similarity_matrix)

        # 对于自监督学习，我们可以将批次中的样本与其增强版本配对
        # 这里假设前半部分和后半部分是配对的增强样本
        for i in range(batch_size // 2):
            pos_mask[i, i + batch_size // 2] = 1
            pos_mask[i + batch_size // 2, i] = 1

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)

        # 正样本对
        pos_pairs = exp_sim * pos_mask
        # 所有样本对（除了自己）
        all_pairs = exp_sim * (1 - torch.eye(batch_size).to(features.device))

        # InfoNCE损失
        loss = -torch.log((pos_pairs.sum(1) + 1e-8) / (all_pairs.sum(1) + 1e-8))
        return loss.mean()


class FashionDatasetWithAugmentation(FashionDataset):
    """带数据增强的时尚数据集"""

    def __init__(self, extractor=None, root='outputs/train/fashiongen_crops', augment=True):
        super().__init__(extractor=extractor, root=root)
        self.augment = augment
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ])
        else:
            self.augmentation = None

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if self.augment and self.augmentation:
            # 获取原始图像
            image_path = self.image_paths[idx]
            original_image = Image.open(image_path).convert("RGB")

            # 应用数据增强
            augmented_image = self.augmentation(original_image)

            # 处理增强图像
            if self.extractor:
                augmented_pixel_values = self.extractor(images=augmented_image, return_tensors="pt")[
                    "pixel_values"].squeeze(0)
            else:
                augmented_pixel_values = self.default_transform(augmented_image)

            # 返回原始和增强的图像
            item['augmented_pixel_values'] = augmented_pixel_values

        return item


def train_vit_lora_optimized():
    # === 加载预训练模型和图像处理器 ===
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # 使用混合精度加载模型，但不直接转换为half()
    base_model = ViTModel.from_pretrained(
        'google/vit-base-patch16-224',
        torch_dtype=torch.float16  # 使用半精度加载权重
    )

    # === LoRA 配置 ===
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        bias="none",
        target_modules=["query", "value"],
        inference_mode=False,
        init_lora_weights=True
    )
    model = get_peft_model(base_model, lora_config)
    model = model.cuda()  # 只移到GPU，不转换为half()

    # === 加载数据（使用带增强的数据集）===
    train_ds = FashionDatasetWithAugmentation(extractor=processor, root='outputs/train/fashiongen_crops', augment=True)
    val_ds = FashionDatasetWithAugmentation(extractor=processor, root='outputs/val/fashiongen_crops',
                                            augment=False)  # 验证集不增强

    # 优化数据加载（充分利用24G显存）
    train_loader = DataLoader(
        train_ds,
        batch_size=64,  # 增大批量大小
        shuffle=True,
        num_workers=8,  # 增加数据加载进程
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,  # 验证集也增大批量
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # 优化优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 使用对比损失
    contrastive_loss_fn = ContrastiveLoss(temperature=0.5)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs("outputs", exist_ok=True)

    # === 训练配置 ===
    num_epochs = 15  # 减少轮次以加快训练
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0

    train_losses = []
    val_losses = []

    print(f"开始训练：{len(train_ds)} 个训练样本，{len(val_ds)} 个验证样本")
    print(f"批量大小: 64, 混合精度训练, 优化数据加载")

    for epoch in range(num_epochs):
        # === 训练阶段 ===
        model.train()
        total_loss = 0
        num_batches = 0

        # 添加更详细的进度信息
        progress_bar = tqdm(train_loader, desc=f"[Train Epoch {epoch}]", ncols=100)

        for batch in progress_bar:
            # 使用非阻塞传输，但保持数据为float16
            x_original = batch['pixel_values'].cuda(non_blocking=True)
            x_augmented = batch['augmented_pixel_values'].cuda(non_blocking=True)

            optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast():
                features_original = model(pixel_values=x_original).last_hidden_state[:, 0, :]
                features_augmented = model(pixel_values=x_augmented).last_hidden_state[:, 0, :]
                all_features = torch.cat([features_original, features_augmented], dim=0)
                loss = contrastive_loss_fn(all_features)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条显示当前损失
            if num_batches % 50 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        # === 验证阶段 ===
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val Epoch {epoch}]", ncols=100, leave=False):
                x = batch['pixel_values'].cuda(non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = model(pixel_values=x).last_hidden_state[:, 0, :]
                    batch_loss = contrastive_loss_fn(output)
                val_loss += batch_loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # 学习率调度
        scheduler.step(avg_val_loss)

        print(f"[Epoch {epoch}] Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # === 模型选择和保存 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'outputs/vit_lora_best.pt')
            print(f"★ 新的最佳模型已保存，验证损失: {best_val_loss:.4f}")
            patience_counter = 0
            model.save_pretrained('outputs/vit_lora_best_peft')
        else:
            patience_counter += 1

        # 每3轮保存检查点（更频繁保存）
        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), f'outputs/vit_lora_epoch_{epoch + 1}.pt')
            model.save_pretrained(f'outputs/vit_lora_peft_epoch_{epoch + 1}')

        if patience_counter >= patience:
            print(f"⚠️ 早停机制触发：连续 {patience} 个epoch验证损失无改善")
            print(f"训练在第 {epoch} 轮停止")
            break

        if epoch > 0:
            train_improvement = train_losses[-2] - train_losses[-1]
            val_improvement = val_losses[-2] - val_losses[-1]
            print(f"  → 训练损失改善: {train_improvement:.4f}, 验证损失改善: {val_improvement:.4f}")

    # === 保存最终模型 ===
    torch.save(model.state_dict(), 'outputs/vit_lora_final.pt')
    model.save_pretrained('outputs/vit_lora_final_peft')
    print("🏁 训练完成")
    print(f"📈 最佳验证损失: {best_val_loss:.4f}")
    print("💾 模型已保存至 outputs/ 目录")

    # === 训练报告 ===
    print("\n=== 训练总结 ===")
    print(f"总训练轮数: {len(train_losses)}")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {val_losses[-1]:.4f}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"损失改善: {train_losses[0] - train_losses[-1]:.4f}")

    return model, processor


if __name__ == "__main__":
    train_vit_lora_optimized()
