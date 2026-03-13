#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高维特征降维可视化工具
用于将高维特征向量降维到2D或3D空间并按类别绘制散点图
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.colors import ListedColormap

def load_data(input_path):
    """加载数据，支持CSV和NPZ格式"""
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
        X = data.iloc[:, :-1].values  # 所有列除了最后一列作为特征
        y = data.iloc[:, -1].values   # 最后一列作为标签
    elif input_path.endswith('.npz'):
        data = np.load(input_path)
        X = data['X']
        y = data['y']
    else:
        raise ValueError("不支持的文件格式，请使用.csv或.npz文件")
    
    return X, y

def apply_dimensionality_reduction(X, method='pca', n_components=2, random_state=42):
    """应用降维方法"""
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("不支持的降维方法，请选择 'pca', 'tsne' 或 'umap'")
    
    X_reduced = reducer.fit_transform(X)
    return X_reduced

def plot_distribution(X_reduced, y, output_path, method, dim=2):
    """绘制降维后的特征分布散点图"""
    # 获取唯一的类别
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    # 设置颜色映射
    if n_classes <= 10:
        cmap = plt.cm.tab10
    elif n_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis
    
    colors = cmap(np.linspace(0, 1, n_classes))
    
    plt.figure(figsize=(10, 8))
    
    if dim == 2:
        # 2D散点图
        for i, cls in enumerate(unique_classes):
            mask = (y == cls)
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                        color=colors[i], label=f'Class {cls}', alpha=0.7)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    
    elif dim == 3:
        # 3D散点图
        ax = plt.figure(figsize=(10, 8)).add_subplot(111, projection='3d')
        
        for i, cls in enumerate(unique_classes):
            mask = (y == cls)
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                      color=colors[i], label=f'Class {cls}', alpha=0.7)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
    
    plt.title(f'Feature Distribution after {method.upper()} Reduction ({dim}D)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {output_path}")
    plt.close()

def generate_demo_data(n_samples=500, n_features=50, n_classes=5, overlap=0.3, random_state=42):
    """生成演示数据，包含5个类别，带有少量重叠"""
    np.random.seed(random_state)
    
    # 为每个类别生成中心点
    centers = np.random.randn(n_classes, n_features) * 10
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # 生成围绕中心点的数据，重叠度由overlap控制
        X_class = np.random.randn(samples_per_class, n_features) * overlap + centers[i]
        y_class = np.ones(samples_per_class) * i
        
        X.append(X_class)
        y.append(y_class)
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    return X, y

def save_demo_data(X, y, output_csv='demo_data.csv', output_npz='demo_data.npz'):
    """保存演示数据为CSV和NPZ格式"""
    # 保存为CSV
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv(output_csv, index=False)
    print(f"演示数据已保存为CSV: {output_csv}")
    
    # 保存为NPZ
    np.savez(output_npz, X=X, y=y)
    print(f"演示数据已保存为NPZ: {output_npz}")

def main():
    parser = argparse.ArgumentParser(description='高维特征降维可视化工具')
    parser.add_argument('--input', type=str, help='输入文件路径 (.csv 或 .npz)')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne', 'umap'], 
                        help='降维方法: pca, tsne, 或 umap')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3], 
                        help='降维后的维度 (2 或 3)')
    parser.add_argument('--output', type=str, default='outputs/feature_distribution.png', 
                        help='输出图像路径')
    parser.add_argument('--demo', action='store_true', help='生成并使用演示数据')
    
    args = parser.parse_args()
    
    # 生成演示数据或加载用户提供的数据
    if args.demo or args.input is None:
        print("生成演示数据...")
        X, y = generate_demo_data()
        
        # 保存演示数据
        save_demo_data(X, y)
    else:
        print(f"从 {args.input} 加载数据...")
        X, y = load_data(args.input)
    
    # 应用降维
    print(f"使用 {args.method} 将数据降维到 {args.dim}D...")
    X_reduced = apply_dimensionality_reduction(X, method=args.method, n_components=args.dim)
    
    # 绘制并保存图像
    print("绘制特征分布图...")
    plot_distribution(X_reduced, y, args.output, args.method, args.dim)

if __name__ == '__main__':
    main()