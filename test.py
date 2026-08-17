import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from ssim import SSIM

def test(model, test_loader, criterion, transformer, model_path, device, save_figures=True, num_visualize=20, save_csv=True, L=5.0):
    result_dir = "./test_results_2d"
    csv_dir = os.path.join(result_dir, "csv_results")
    os.makedirs(result_dir, exist_ok=True)
    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    criterion = criterion.to(device)

    # 初始化评价指标
    ms_ssim_similarity = SSIM(
        data_dim=2,
        in_channels=1,
        window_size=11,
        sigma=1.5,
        L=L,
        ensemble_kernel=True
    ).to(device)
    l1_criterion = torch.nn.L1Loss()
    l2_criterion = torch.nn.MSELoss()
    ssim_lambda = checkpoint.get('ssim_lambda', 0)

    print(f"\n=== 测试模型信息 ===")
    print(f"模型路径: {model_path}")
    print(f"运行设备: {device}")
    print(f"训练轮次: {checkpoint.get('epoch', '未知')}")
    print(f"MS-SSIM正则化系数: {ssim_lambda:.8f}")
    print(f"====================\n")

    # 初始化损失累计
    total_smooth_l1_loss = total_l1_loss = total_l2_loss = total_ssim_loss = 0.0
    total_input_smooth_l1_loss = total_input_l1_loss = total_input_l2_loss = total_input_ssim_loss = 0.0
    all_outputs = []
    all_labels = []
    all_inputs = []

    with torch.no_grad():
        loop = tqdm(test_loader, total=len(test_loader), leave=True)
        loop.set_description(f"Testing | Device: {device}")

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # 计算输出vs标签损失
            smooth_l1_loss = criterion(outputs, labels)
            l1_loss = l1_criterion(outputs, labels)
            l2_loss = l2_criterion(outputs, labels)
            ssim_loss = 1.0 - ms_ssim_similarity(outputs, labels)

            # 计算输入vs标签损失
            input_smooth_l1_loss = criterion(inputs, labels)
            input_l1_loss = l1_criterion(inputs, labels)
            input_l2_loss = l2_criterion(inputs, labels)
            input_ssim_loss = 1.0 - ms_ssim_similarity(inputs, labels)

            # 累计损失
            total_smooth_l1_loss += smooth_l1_loss.item()
            total_l1_loss += l1_loss.item()
            total_l2_loss += l2_loss.item()
            total_ssim_loss += ssim_loss.item()

            total_input_smooth_l1_loss += input_smooth_l1_loss.item()
            total_input_l1_loss += input_l1_loss.item()
            total_input_l2_loss += input_l2_loss.item()
            total_input_ssim_loss += input_ssim_loss.item()

            # 收集数据
            all_inputs.append(inputs.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            loop.set_postfix(
                smooth_l1=f"{smooth_l1_loss.item():.6f}",
                ssim_loss=f"{ssim_loss.item():.6f}"
            )

    # 计算平均损失
    avg_smooth_l1_loss = total_smooth_l1_loss / len(test_loader)
    avg_l1_loss = total_l1_loss / len(test_loader)
    avg_l2_loss = total_l2_loss / len(test_loader)
    avg_ssim_loss = total_ssim_loss / len(test_loader)
    avg_input_smooth_l1_loss = total_input_smooth_l1_loss / len(test_loader)
    avg_input_l1_loss = total_input_l1_loss / len(test_loader)
    avg_input_l2_loss = total_input_l2_loss / len(test_loader)
    avg_input_ssim_loss = total_input_ssim_loss / len(test_loader)
    avg_total_loss = avg_smooth_l1_loss + ssim_lambda * avg_ssim_loss

    # 打印评价指标
    print(f"\n=== 测试集性能总结 ===")
    print(f"平均总损失 (SmoothL1 + SSIM损失*λ): {avg_total_loss:.6f}")
    print("\n--- 输出 vs 标签 ---")
    print(f"平均SmoothL1损失：{avg_smooth_l1_loss:.6f}")
    print(f"平均L1损失（MAE）：{avg_l1_loss:.6f}")
    print(f"平均L2损失（MSE）：{avg_l2_loss:.6f}")
    print(f"平均SSIM损失：{avg_ssim_loss:.6f}")
    print("\n--- 输入 vs 标签 ---")
    print(f"平均SmoothL1损失：{avg_input_smooth_l1_loss:.6f}")
    print(f"平均L1损失：{avg_input_l1_loss:.6f}")
    print(f"平均L2损失：{avg_input_l2_loss:.6f}")
    print(f"平均SSIM损失：{avg_input_ssim_loss:.6f}")

    # 逆归一化并计算原始空间损失
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_inputs_original = transformer.inverse_transform(all_inputs)
    all_outputs_original = np.abs(transformer.inverse_transform(all_outputs))
    all_labels_original = transformer.inverse_transform(all_labels)

    # 原始空间损失
    mse_input_label = np.mean((all_inputs_original - all_labels_original) ** 2)
    mse_output_label = np.mean((all_outputs_original - all_labels_original) ** 2)
    mae_input_label = np.mean(np.abs(all_inputs_original - all_labels_original))
    mae_output_label = np.mean(np.abs(all_outputs_original - all_labels_original))

    print(f"\n=== 原始空间损失 ===")
    print(f"输入 vs 标签：MAE={mae_input_label:.6f}, MSE={mse_input_label:.6f}")
    print(f"输出 vs 标签：MAE={mae_output_label:.6f}, MSE={mse_output_label:.6f}")

    # 可视化
    if save_figures:
        max_samples = min(num_visualize, all_inputs_original.shape[0])
        for i in range(max_samples):
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(all_inputs_original[i, 0], cmap='jet', aspect=0.04)
            plt.title(f'Input (sample {i + 1})')
            plt.colorbar(label='Value')

            plt.subplot(1, 3, 2)
            plt.imshow(all_outputs_original[i, 0], cmap='jet', aspect=0.04)
            plt.title(f'Output (sample {i + 1})')
            plt.colorbar(label='Value')

            plt.subplot(1, 3, 3)
            plt.imshow(all_labels_original[i, 0], cmap='jet', aspect=0.04)
            plt.title(f'Label (sample {i + 1})')
            plt.colorbar(label='Value')

            plt.suptitle(f'Sample {i + 1} Comparison', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f'sample_{i + 1}_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        print(f"可视化结果已保存至: {result_dir}")

    # 保存CSV
    if save_csv:
        def reshape_to_df(data):
            M, _, H, N = data.shape
            return pd.DataFrame(data.transpose(0, 3, 2, 1).reshape(-1, H))

        reshape_to_df(all_inputs_original).to_csv(os.path.join(csv_dir, "input_test_results.csv"), index=False, header=False)
        reshape_to_df(all_outputs_original).to_csv(os.path.join(csv_dir, "output_test_results.csv"), index=False, header=False)
        reshape_to_df(all_labels_original).to_csv(os.path.join(csv_dir, "label_test_results.csv"), index=False, header=False)
        print(f"CSV结果已保存至: {csv_dir}")

    return {
        "total_loss": avg_total_loss,
        "output_smooth_l1": avg_smooth_l1_loss,
        "output_l1": avg_l1_loss,
        "output_l2": avg_l2_loss,
        "output_ssim_loss": avg_ssim_loss,
        "input_l1": avg_input_l1_loss,
        "input_l2": avg_input_l2_loss,
        "input_ssim_loss": avg_input_ssim_loss,
        "original_input_mae": mae_input_label,
        "original_input_mse": mse_input_label,
        "original_output_mae": mae_output_label,
        "original_output_mse": mse_output_label
    }