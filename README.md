# Unet网络

本仓库按原始 `unet_2d.zip` 的目录结构上传了代码、模型、配置、测试结果和可上传的数据文件。

## 目录说明

- `main.py`、`train.py`、`test.py`：训练与测试入口
- `dataset.py`、`loss.py`、`ssim.py`：数据集与损失函数代码
- `unet/`、`model/`：网络代码
- `data/`：数据文件（超大原始数据集除外）
- `saved_models_2d/`：模型权重与训练曲线
- `test_results_2d/`：测试结果

## 未上传文件

`data/dataset_GNI.mat` 原文件约 1 GB，超过 GitHub 普通仓库单文件 100 MB 限制，因此未提交。原始完整压缩包仍保存在本地：`unet_2d.zip`。
