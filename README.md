# 高维特征降维可视化工具

这个工具用于将高维特征向量降维到2D或3D空间，并按类别绘制散点图，以便可视化不同类别在特征空间中的分布情况。

## 说明

这个工具生成的是**高维特征降维后的可视化分布图**，而不是原始信号图。它主要用于：

- 可视化神经网络提取的embedding/feature vector在降维后的分布
- 观察不同类别在特征空间中的聚类情况和分离程度
- 评估特征提取方法的有效性

通常用于机器学习和深度学习中，帮助理解模型提取的特征如何区分不同类别的样本。

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python plot_feature_distribution.py --input data.csv --method tsne --dim 2 --output outputs/tsne_2d.png
```

### 参数说明

- `--input`: 输入文件路径，支持CSV格式（前几列是特征，最后一列是标签）或NPZ格式（包含X和y数组）
- `--method`: 降维方法，可选 'pca', 'tsne', 或 'umap'
- `--dim`: 降维后的维度，可选 2 或 3
- `--output`: 输出图像路径
- `--demo`: 生成并使用演示数据（5类，带少量重叠）

### 示例

```bash
# 使用PCA降维到2D
python plot_feature_distribution.py --input data.csv --method pca --dim 2 --output outputs/pca_2d.png

# 使用t-SNE降维到3D
python plot_feature_distribution.py --input data.csv --method tsne --dim 3 --output outputs/tsne_3d.png

# 使用UMAP降维到2D
python plot_feature_distribution.py --input data.csv --method umap --dim 2 --output outputs/umap_2d.png

# 生成演示数据并使用
python plot_feature_distribution.py --demo --method tsne --dim 3 --output outputs/demo_tsne_3d.png
```

## 输入格式

### CSV格式
CSV文件中，前几列应为特征数据，最后一列应为标签（类别）。

### NPZ格式
NPZ文件应包含两个数组：
- `X`: 特征矩阵，形状为 (n_samples, n_features)
- `y`: 标签向量，形状为 (n_samples,)

## 输出

工具将生成一个PNG格式的散点图，其中：
- 不同颜色代表不同类别
- 2D或3D视图展示降维后的特征分布
- 图像包含图例和标题
- 保存到指定的输出路径
