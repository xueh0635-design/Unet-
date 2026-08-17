import torch
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import ResistivityDataset2D, ResistivityPreprocessor2D
from unet import UNet2D
from train import train
from test import test

if __name__ == "__main__":
    # 设备初始化
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 核心参数配置
    mode = "test"  # "train" or "test"
    model_dir = "./saved_models_2d"
    mat_path = "./data/dataset_GNI.mat"
    sample_key = 'input_GNI_all'
    label_key = 'gt_all'
    split_width = 8
    epochs = 100
    batch_size = 256
    steps_per_batch = 1
    initial_lr = 1e-3
    lr_decay_epochs = 40
    lr_decay_factor = 0.5
    min_learning_rate = 1e-7

    # 创建模型保存目录
    os.makedirs(model_dir, exist_ok=True)

    # 初始化预处理工具
    preprocessor = ResistivityPreprocessor2D(use_log=False, epsilon=1e-8)

    # 加载完整数据集（未预处理）
    full_dataset = ResistivityDataset2D(
        mat_path=mat_path,
        sample_key=sample_key,
        label_key=label_key,
        preprocessor=None,
        split_width=split_width
    )
    dataset_size = len(full_dataset)
    print(f"分割后总样本数: {dataset_size}（原始样本按{split_width}宽度分割）")

    # 划分训练集/验证集索引
    train_indices, val_indices = train_test_split(
        list(range(dataset_size)),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 用训练集拟合预处理参数
    train_samples = full_dataset.split_samples[train_indices]
    preprocessor.fit(train_samples)

    # 创建训练集和验证集
    train_dataset = Subset(
        ResistivityDataset2D(
            mat_path=mat_path,
            sample_key=sample_key,
            label_key=label_key,
            preprocessor=preprocessor,
            split_width=split_width
        ),
        train_indices
    )
    val_dataset = Subset(
        ResistivityDataset2D(
            mat_path=mat_path,
            sample_key=sample_key,
            label_key=label_key,
            preprocessor=preprocessor,
            split_width=split_width
        ),
        val_indices
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 训练或测试
    if mode == "train":
        model = UNet2D(
            in_channels=1,
            out_classes=1,
            normalization="instance",
            residual=True,
            preactivation=True,
            activation="LeakyReLU",
            upsampling_type="conv",
            padding_mode="replicate",
            num_encoding_blocks = 4
        )
        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

        # 计算SSIM的L值
        L = float(torch.ceil(torch.tensor(train_samples).max() - torch.tensor(train_samples).min()))

        # 调用训练函数
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            steps_per_batch=steps_per_batch,
            model_dir=model_dir,
            transformer=preprocessor,
            device=device,
            L=L,
            ssim_lambda=1.0,
            tv_lambda=0.0,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_factor=lr_decay_factor,
            min_lr=min_learning_rate
        )

    elif mode == "test":
        # 构建测试集（使用自身预处理参数）
        test_dataset_raw = ResistivityDataset2D(
            mat_path=mat_path,
            sample_key=sample_key,
            label_key=label_key,
            preprocessor=None,
            split_width=split_width
        )
        test_preprocessor = ResistivityPreprocessor2D(use_log=False)
        test_preprocessor.fit(test_dataset_raw.split_samples)
        print(f"测试集参数：mean={test_preprocessor.mean}, std={test_preprocessor.std}")

        test_dataset = ResistivityDataset2D(
            mat_path=mat_path,
            sample_key=sample_key,
            label_key=label_key,
            preprocessor=test_preprocessor,
            split_width=split_width
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = UNet2D(
            in_channels=1,
            out_classes=1,
            normalization="instance",
            residual=True,
            preactivation=True,
            activation="LeakyReLU",
            upsampling_type="conv",
            padding_mode="replicate",
            num_encoding_blocks=4
        )

        # 测试模型
        best_model_path = os.path.join(model_dir, "best_model_2d.pth")
        if not os.path.exists(best_model_path):
            print(f"错误：未找到模型文件 {best_model_path}")
        else:
            L = float(torch.ceil(torch.tensor(test_dataset_raw.split_samples).max() - torch.tensor(test_dataset_raw.split_samples).min()))
            test(
                model=model,
                test_loader=test_loader,
                criterion=torch.nn.SmoothL1Loss(),
                transformer=test_preprocessor,
                model_path=best_model_path,
                device=device,
                L=L
            )

    else:
        print(f"错误：未知模式 '{mode}'，请选择 'train' 或 'test'")