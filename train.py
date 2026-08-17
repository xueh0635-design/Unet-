import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from ssim import SSIM
from loss import tv_loss_2d

def train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs,
        steps_per_batch,
        model_dir,
        transformer,
        device,
        L,
        ssim_lambda=0.1,
        tv_lambda=0.0,
        lr_decay_epochs=20,
        lr_decay_factor=0.5,
        min_lr=1e-7
):
    # 初始化SSIM模块
    ms_ssim_module = SSIM(
        data_dim=2,
        in_channels=1,
        window_size=11,
        sigma=1.5,
        L=L,
        ensemble_kernel=True
    ).to(device)

    # 损失记录
    loss_records = {
        'train_total': [], 'train_smooth_l1': [], 'train_ms_ssim': [], 'train_tv': [],
        'val_total': [], 'val_smooth_l1': [], 'val_ms_ssim': [], 'val_tv': []
    }
    learning_rates = []

    model = model.to(device)
    criterion = criterion.to(device)
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir, "best_model_2d.pth")
    patience = 50
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        train_total = train_smooth_l1 = train_ms_ssim = train_tv = 0.0
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        train_loop.set_description(f"Epoch [{epoch + 1}/{epochs}] (Train)")

        for inputs, labels in train_loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            batch_total = batch_smooth_l1 = batch_ms_ssim = batch_tv = 0.0
            for _ in range(steps_per_batch):
                optimizer.zero_grad()
                outputs = model(inputs)

                smooth_l1_loss = criterion(outputs, labels)
                ssim_loss = 1.0 - ms_ssim_module(outputs, labels)
                tv_reg = tv_loss_2d(outputs)
                total_loss = smooth_l1_loss + ssim_lambda * ssim_loss + tv_lambda * tv_reg
                total_loss.backward()
                optimizer.step()

                batch_total += total_loss.item()
                batch_smooth_l1 += smooth_l1_loss.item()
                batch_ms_ssim += ssim_loss.item()
                batch_tv += tv_reg.item()

            # 计算批次平均损失
            avg_total = batch_total / steps_per_batch
            avg_smooth_l1 = batch_smooth_l1 / steps_per_batch
            avg_ssim = batch_ms_ssim / steps_per_batch
            avg_tv = batch_tv / steps_per_batch

            # 累加 epoch 损失
            train_total += avg_total
            train_smooth_l1 += avg_smooth_l1
            train_ms_ssim += avg_ssim
            train_tv += avg_tv

            train_loop.set_postfix(
                total=f"{avg_total:.6f}",
                smooth_l1=f"{avg_smooth_l1:.6f}",
                ssim=f"{avg_ssim:.6f}",
                tv=f"{avg_tv:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )

        # 计算 epoch 平均损失
        epoch_train_total = train_total / len(train_loader)
        epoch_train_smooth_l1 = train_smooth_l1 / len(train_loader)
        epoch_train_ssim = train_ms_ssim / len(train_loader)
        epoch_train_tv = train_tv / len(train_loader)

        # 验证阶段
        model.eval()
        val_total = val_smooth_l1 = val_ssim = val_tv = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                smooth_l1_loss = criterion(outputs, labels)
                ssim_loss = 1.0 - ms_ssim_module(outputs, labels)
                tv_reg = tv_loss_2d(outputs)
                total_loss = smooth_l1_loss + ssim_lambda * ssim_loss + tv_lambda * tv_reg

                val_total += total_loss.item()
                val_smooth_l1 += smooth_l1_loss.item()
                val_ssim += ssim_loss.item()
                val_tv += tv_reg.item()

        # 计算验证集平均损失
        epoch_val_total = val_total / len(val_loader)
        epoch_val_smooth_l1 = val_smooth_l1 / len(val_loader)
        epoch_val_ssim = val_ssim / len(val_loader)
        epoch_val_tv = val_tv / len(val_loader)

        model.train()

        # 记录损失
        loss_records['train_total'].append(epoch_train_total)
        loss_records['train_smooth_l1'].append(epoch_train_smooth_l1)
        loss_records['train_ms_ssim'].append(epoch_train_ssim)
        loss_records['train_tv'].append(epoch_train_tv)
        loss_records['val_total'].append(epoch_val_total)
        loss_records['val_smooth_l1'].append(epoch_val_smooth_l1)
        loss_records['val_ms_ssim'].append(epoch_val_ssim)
        loss_records['val_tv'].append(epoch_val_tv)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 打印 epoch 信息
        print(
            f"\nEpoch {epoch+1}/{epochs} | "
            f"Train: Total={epoch_train_total:.6f}, SmoothL1={epoch_train_smooth_l1:.6f}, "
            f"SSIM={epoch_train_ssim:.6f}, TV={epoch_train_tv:.6f} | "
            f"Val: Total={epoch_val_total:.6f}, SmoothL1={epoch_val_smooth_l1:.6f}, "
            f"SSIM={epoch_val_ssim:.6f}, TV={epoch_val_tv:.6f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        # 保存最佳模型
        if epoch_val_total < best_val_loss:
            best_val_loss = epoch_val_total
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'mean': transformer.mean,
                'std': transformer.std,
                'ssim_lambda': ssim_lambda,
                'tv_lambda': tv_lambda,
                'L': L
            }, best_model_path)
            print(f"✓ 最佳模型已保存（验证损失: {best_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发！最佳验证损失: {best_val_loss:.6f}")
                break

        # 学习率衰减
        if (epoch + 1) % lr_decay_epochs == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr > min_lr:
                new_lr = max(current_lr * lr_decay_factor, min_lr)
                optimizer.param_groups[0]['lr'] = new_lr
                print(f"学习率调整为: {new_lr:.6f}")

    # 绘制损失曲线
    plot_loss_curves(loss_records, learning_rates, model_dir)
    print("训练完成！")


def plot_loss_curves(loss_records, learning_rates, save_dir):
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(loss_records['train_total'], label='Train Total Loss')
    plt.plot(loss_records['val_total'], label='Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss (SmoothL1 + SSIM)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(loss_records['train_smooth_l1'], label='Train SmoothL1')
    plt.plot(loss_records['val_smooth_l1'], label='Validation SmoothL1')
    plt.xlabel('Epoch')
    plt.ylabel('SmoothL1 Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(loss_records['train_ms_ssim'], label='Train SSIM Loss')
    plt.plot(loss_records['val_ms_ssim'], label='Validation SSIM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves_train_val_2d.png'), dpi=300)
    plt.close()

    # 学习率曲线
    plt.figure(figsize=(10, 4))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate_curve_2d.png'), dpi=300)
    plt.close()