#!/usr/bin/env python3
"""
可视化BloodMNIST数据集的前十个样本
显示图像和对应的文本描述
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_bloodmnist_data(data_path):
    """加载BloodMNIST数据"""
    print(f"正在加载数据从: {data_path}")
    
    # 加载图像数据
    images_path = os.path.join(data_path, "bloodmnist_images.pkl")
    text_path = os.path.join(data_path, "bloodmnist_text_descriptions.json")
    
    # 检查文件是否存在
    if not os.path.exists(images_path):
        print(f"警告: 图像文件不存在 {images_path}")
        images = None
        labels = None
    else:
        try:
            with open(images_path, 'rb') as f:
                image_data = pickle.load(f)
            
            if isinstance(image_data, dict):
                if 'images' in image_data:
                    images = image_data['images']
                    labels = image_data.get('labels', None)
                else:
                    # 如果pickle文件结构不同，尝试其他方式
                    images = image_data
                    labels = None
            else:
                images = image_data
                labels = None
            
            # 转换为numpy数组如果是列表
            if isinstance(images, list):
                images = np.array(images)
                print(f"图像数据类型: 列表 -> numpy数组")
            
            print(f"图像数据形状: {images.shape if hasattr(images, 'shape') else f'列表长度: {len(images)}'}")
            print(f"图像数据类型: {type(images)}")
            if hasattr(images, 'shape') and len(images) > 0:
                print(f"单个图像形状: {images[0].shape if hasattr(images[0], 'shape') else type(images[0])}")
            
        except Exception as e:
            print(f"加载图像数据失败: {e}")
            images = None
            labels = None
    
    # 加载文本数据
    if not os.path.exists(text_path):
        print(f"错误: 文本文件不存在 {text_path}")
        return None, None, None
    
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        
        # 提取样本数据
        samples = text_data.get('data', text_data.get('samples', []))
        
        print(f"找到 {len(samples)} 个文本样本")
        print(f"文本数据结构: {list(samples[0].keys()) if samples else '无数据'}")
        
        return images, labels, samples
        
    except Exception as e:
        print(f"加载文本数据失败: {e}")
        return None, None, None

def visualize_samples(images, labels, text_samples, num_samples=10):
    """可视化前N个样本"""
    
    # 确定要显示的样本数量
    max_samples = min(num_samples, len(text_samples))
    if images is not None:
        max_samples = min(max_samples, len(images))
    
    print(f"将显示前 {max_samples} 个样本")
    
    # 创建子图
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle('BloodMNIST 数据集前10个样本', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # 类别名称映射
    class_names = [
        "basophil", "eosinophil", "erythroblast", "immature_granulocytes",
        "lymphocyte", "monocyte", "neutrophil", "platelet"
    ]
    
    class_names_zh = [
        "嗜碱性粒细胞", "嗜酸性粒细胞", "红细胞母细胞", "幼稚粒细胞",
        "淋巴细胞", "单核细胞", "中性粒细胞", "血小板"
    ]
    
    for i in range(max_samples):
        ax = axes[i]
        
        # 获取文本样本信息
        sample = text_samples[i]
        sample_id = sample.get('sample_id', i)
        label = sample.get('label', 0)
        class_name = sample.get('class_name', 'unknown')
        text_desc = sample.get('text_description', '无描述')
        
        # 显示图像
        if images is not None and i < len(images):
            img = images[i]
            
            # 确保img是numpy数组
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            # 处理图像数据格式
            if len(img.shape) == 3:
                if img.shape[0] == 3:  # CHW格式
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[2] == 3:  # HWC格式
                    pass
                else:
                    img = img[:, :, 0]  # 取第一个通道
            elif len(img.shape) == 2:  # 灰度图
                pass
            
            # 归一化图像到[0,1]
            if img.max() > 1.0:
                img = img / 255.0
            
            # 确保值在合理范围内
            img = np.clip(img, 0, 1)
            
            # 如果是彩色图像
            if len(img.shape) == 3 and img.shape[2] == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
        else:
            # 如果没有图像，显示占位符
            ax.text(0.5, 0.5, f'样本 {sample_id}\n无图像数据', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # 设置标题和标签
        zh_class = class_names_zh[label] if label < len(class_names_zh) else class_name
        title = f'ID:{sample_id} | {zh_class}\n({class_name})'
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # 在图像下方显示文本描述
        wrapped_text = text_desc[:50] + '...' if len(text_desc) > 50 else text_desc
        ax.text(0.5, -0.1, wrapped_text, ha='center', va='top', 
               transform=ax.transAxes, fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 隐藏空白子图
    for i in range(max_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)
    
    return fig

def print_detailed_info(text_samples, num_samples=10):
    """打印详细的文本信息"""
    print("\n" + "="*80)
    print("详细样本信息")
    print("="*80)
    
    max_samples = min(num_samples, len(text_samples))
    
    for i in range(max_samples):
        sample = text_samples[i]
        print(f"\n样本 {i+1}:")
        print(f"  ID: {sample.get('sample_id', 'N/A')}")
        print(f"  标签: {sample.get('label', 'N/A')}")
        print(f"  类别: {sample.get('class_name', 'N/A')}")
        print(f"  图像形状: {sample.get('image_shape', 'N/A')}")
        print(f"  文本描述: {sample.get('text_description', 'N/A')}")
        
        # 如果有图像特征信息
        if 'image_features' in sample:
            features = sample['image_features']
            print(f"  图像特征:")
            for key, value in features.items():
                print(f"    {key}: {value:.4f}")
        
        print("-" * 60)

def main():
    """主函数"""
    data_path = r"F:\Desktop\HyperBridge\HyperBridge-main\multimodal_medmnist_datasets\bloodmnist"
    
    print("BloodMNIST 数据集可视化工具")
    print(f"数据路径: {data_path}")
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据路径不存在 {data_path}")
        return
    
    # 加载数据
    images, labels, text_samples = load_bloodmnist_data(data_path)
    
    if text_samples is None:
        print("错误: 无法加载数据")
        return
    
    # 打印数据集基本信息
    print(f"\n数据集信息:")
    print(f"  文本样本数量: {len(text_samples)}")
    if images is not None:
        if hasattr(images, 'shape'):
            print(f"  图像数据形状: {images.shape}")
        else:
            print(f"  图像数据: 列表，长度为 {len(images)}")
            if len(images) > 0:
                sample_img = images[0]
                if hasattr(sample_img, 'shape'):
                    print(f"  单个图像形状: {sample_img.shape}")
                else:
                    print(f"  单个图像类型: {type(sample_img)}")
    else:
        print(f"  图像数据: 未找到或加载失败")
    
    # 可视化前10个样本
    try:
        fig = visualize_samples(images, labels, text_samples, num_samples=10)
        
        # 保存图像
        output_path = os.path.join(data_path, "visualization_top10.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存到: {output_path}")
        
        # 显示图像
        plt.show()
        
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 打印详细信息
    print_detailed_info(text_samples, num_samples=10)
    
    print("\n可视化完成!")

if __name__ == "__main__":
    main()
