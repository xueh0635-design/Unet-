2D U-Net 电阻率数据处理项目
该项目基于 2D U-Net 模型实现电阻率数据的处理与重建，支持训练、测试流程，包含数据预处理、模型训练、性能评估及结果可视化功能。

环境配置
1.克隆或下载项目到本地
2.安装依赖库：
pip install -r requirements.txt

使用方法
1. 数据准备
准备 MAT 格式的数据集（支持 v7.3+），包含输入数据和标签数据
数据集路径及键名需在main.py中配置（mat_path、sample_key、label_key参数）

2. 模型训练
打开main.py，设置mode = "train"
配置核心参数（可根据需求调整）：

mode = "train"               # 训练模式
mat_path = "./data/dataset.mat"  # 数据集路径   ，注：dataset_GNI.mat为训练数据，包含'input_GNI_all'和'input_all'两种sample_key
split_width = 8              # 子样本分割宽度
epochs = 100                 # 训练轮次
batch_size = 256             # 批次大小
initial_lr = 1e-3            # 初始学习率

3.训练过程：
模型会自动划分训练集（80%）和验证集（20%）
最佳模型会保存至saved_models_2d/best_model_2d.pth
损失曲线和学习率曲线会保存至saved_models_2d/目录


4.模型测试
打开main.py，设置mode = "test"
确保测试数据集路径正确（mat_path）#注：dataset_GNI_test.mat为测试数据，包含'input_GNI_all'和'input_all'两种sample_key；
			        #注：modified_main_file.mat为GN反演结果数据，包含'input_all'一种sample_key；

5.测试结果：
评估指标（MAE、MSE、SSIM 损失等）会打印到控制台
对比图像会保存至test_results_2d/目录
原始数据（输入 / 输出 / 标签）会保存为 CSV 文件，路径：test_results_2d/csv_results/

核心参数说明
参数名称	                                作用说明	                                                                   当前值
split_width	                子样本分割宽度（输入数据宽度方向的分割尺寸）	   8
epochs	                                训练总轮次	                                                   100
batch_size	                                训练批次大小（根据 GPU 内存调整）	                   64-256
initial_lr	                                初始学习率	                                                   1e-3
ssim_lambda	                SSIM 损失的权重系数	                                                   1.0
tv_lambda	                                总变差损失的权重系数	                                    0.0
num_encoding_blocks	U-Net 编码器块数量（在模型实例化时设置）	    4