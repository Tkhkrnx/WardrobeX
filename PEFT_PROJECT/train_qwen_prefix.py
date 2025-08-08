import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model, PeftModel
from data.text_dataset import PromptDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json


def train(resume_from_checkpoint=None):
    model_name = 'Qwen3-0.6B'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查是否从检查点恢复训练
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_cache=False
        ).cuda()

        # 加载PEFT模型
        model = PeftModel.from_pretrained(base_model, resume_from_checkpoint)
        base_model.config.use_cache = False
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_cache=False
        ).cuda()
        base_model.config.use_cache = False

        # Prompt Tuning 配置
        prompt_cfg = PromptTuningConfig(
            task_type='CAUSAL_LM',
            num_virtual_tokens=20,
            prompt_tuning_init="RANDOM",
            tokenizer_name_or_path=model_name
        )
        model = get_peft_model(base_model, prompt_cfg)

    model.print_trainable_parameters()

    # 数据集与 DataLoader
    train_dataset = PromptDataset(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 尝试稍微提高学习率以加快收敛
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # 如果从检查点恢复，尝试加载优化器状态和训练状态
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0

    if resume_from_checkpoint and os.path.exists(os.path.join(resume_from_checkpoint, "training_state.json")):
        training_state = json.load(open(os.path.join(resume_from_checkpoint, "training_state.json"), "r"))
        start_epoch = training_state.get("epoch", 0) + 1
        best_loss = training_state.get("best_loss", float('inf'))
        patience_counter = training_state.get("patience_counter", 0)
        print(f"Resuming from epoch {start_epoch} with best_loss {best_loss:.4f}")

    num_epochs = 500  # 增加训练轮数
    patience = 20  # 增加耐心值

    gradient_accumulation_steps = 4  # 等效 batch size = 2 * 4 = 8
    global_step = 0

    os.makedirs("outputs", exist_ok=True)

    # 记录初始学习率
    print(f"Starting training with learning rate: {3e-5}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0

        # 添加进度条描述，显示当前loss
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Loss: N/A")

        for step, batch in enumerate(progress_bar):
            # 编码输入和输出
            inputs = tokenizer(
                batch['input'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            )
            targets = tokenizer(
                batch['output'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            )
            # 拼接
            input_ids = torch.cat([inputs.input_ids, targets.input_ids], dim=1).to('cuda')
            attention_mask = torch.cat([inputs.attention_mask, targets.attention_mask], dim=1).to('cuda')
            labels = input_ids.clone()
            labels[:, :inputs.input_ids.size(1)] = -100  # 仅对输出部分计算 loss

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * gradient_accumulation_steps

            # 更新进度条描述
            progress_bar.set_description(f"Epoch {epoch} - Loss: {loss.item() * gradient_accumulation_steps:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - avg_loss: {avg_loss:.4f}")

        # 保存检查点（每10个epoch）
        if epoch % 10 == 0:
            checkpoint_path = f'outputs/qwen3_0.6b_prompt_epoch_{epoch}'
            model.save_pretrained(checkpoint_path)
            # 保存训练状态
            training_state = {
                "epoch": epoch,
                "best_loss": best_loss,
                "patience_counter": patience_counter
            }
            with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
                json.dump(training_state, f)
            print(f"Checkpoint saved at epoch {epoch}")

        # 早停与保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained('outputs/qwen3_0.6b_prompt_best')
            print(f"Saved new best model (loss: {best_loss:.4f})")
            patience_counter = 0

            # 如果loss已经降到1.0以下，可以提前保存一个检查点
            if avg_loss < 1.0:
                model.save_pretrained(f'outputs/qwen3_0.6b_prompt_loss_below_1')
                print(f"Model with loss < 1.0 saved")
        else:
            patience_counter += 1

        # 更新训练状态文件
        training_state = {
            "epoch": epoch,
            "best_loss": best_loss,
            "patience_counter": patience_counter
        }
        with open("outputs/qwen3_0.6b_prompt_best/training_state.json", "w") as f:
            json.dump(training_state, f)

        if patience_counter >= patience:
            print(f"No improvement for {patience} epochs, stopping early.")
            break

    # 最终保存
    model.save_pretrained('outputs/qwen3_0.6b_prompt_final')
    print(f"Training completed. Best loss: {best_loss:.4f}")


def resume_training(checkpoint_path):
    """从检查点恢复训练的便捷函数"""
    train(resume_from_checkpoint=checkpoint_path)


if __name__ == '__main__':
    train()
