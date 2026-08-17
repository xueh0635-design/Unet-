# Unet网络

本仓库按原始 `unet_2d.zip` 的目录结构上传了代码、模型、配置、测试结果和可上传的数据文件。

## 目录说明

- `main.py`、`train.py`、`test.py`：训练与测试入口
- `dataset.py`、`loss.py`、`ssim.py`：数据集与损失函数代码
- `unet/`、`model/`：网络代码
- `data/`：数据文件（超大原始数据集除外）
- `saved_models_2d/`：模型权重与训练曲线
- `test_results_2d/`：测试结果

## 大数据集下载与还原

原始 `data/dataset_GNI.mat` 约 1 GB，已切分为 `data/dataset_GNI_parts/` 中的 12 个分片，每个分片约 90 MiB，均低于 GitHub 单文件限制。下载仓库后，在项目根目录运行：

```bash
python3 data/reassemble_dataset_GNI.py
```

脚本会按编号顺序合并分片，并使用 `dataset_GNI.mat.sha256` 校验还原结果。还原后的文件会生成在 `data/dataset_GNI.mat`。
