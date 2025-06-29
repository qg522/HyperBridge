#!/usr/bin/env python3
"""
多数据集超图可视化演示脚本
Demo script for multi-dataset hypergraph visualization

使用方法 Usage:
1. 运行所有三个数据集的对比分析：
   python demo_multi_hypergraph.py

2. 只运行单个数据集（PathMNIST）：
   python demo_multi_hypergraph.py --single

3. 自定义参数运行：
   python demo_multi_hypergraph.py --datasets pathmnist bloodmnist --samples 128
"""

import sys
import os
import argparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import (
    HypergraphConverter, 
    MultiDatasetHypergraphVisualizer,
    visualize_multiple_datasets,
    visualize_hypergraph_for_pathmnist,
    load_medmnist_dataset,
    load_synthetic_dataset
)
import torch
import pickle


def demo_single_dataset():
    """演示单个数据集的超图可视化"""
    print("🔍 Demo: Single Dataset Hypergraph Visualization")
    print("="*60)
    
    visualize_hypergraph_for_pathmnist()


def demo_custom_datasets(dataset_names=['pathmnist', 'bloodmnist'], num_samples=256):
    """演示自定义数据集组合的超图可视化"""
    print(f"🔍 Demo: Custom Datasets Hypergraph Visualization")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Samples per dataset: {num_samples}")
    print("="*60)
    
    # 设置保存目录
    save_dir = "custom_demo_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化转换器和可视化器
    converter = HypergraphConverter(
        similarity_threshold=0.7, 
        max_hyperedge_size=12, 
        feature_dim=64
    )
    visualizer = MultiDatasetHypergraphVisualizer()
    
    hypergraph_datasets = {}
    
    for dataset_name in dataset_names:
        try:
            print(f"\n📊 Processing {dataset_name.upper()}...")
            
            if dataset_name.lower() == 'synthetic':
                images, labels = load_synthetic_dataset(num_samples=num_samples)
            else:
                images, labels = load_medmnist_dataset(dataset_name, num_samples=num_samples)
            
            if images.dtype != torch.float32:
                images = images.float()
            
            hypergraph_data = converter.convert_to_hypergraph(images, labels)
            hypergraph_datasets[dataset_name.upper()] = hypergraph_data
            
            print(f"✅ {dataset_name.upper()}: {hypergraph_data['n_nodes']} nodes, "
                  f"{hypergraph_data['n_hyperedges']} hyperedges")
            
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {str(e)}")
    
    # 创建对比可视化
    if hypergraph_datasets:
        print(f"\n🎨 Creating comparative visualizations...")
        visualizer.create_comparative_visualization(hypergraph_datasets, save_dir)
        
        print(f"\n✅ Demo completed! Results saved in '{save_dir}' directory")
    else:
        print("❌ No datasets were successfully processed.")


def demo_all_datasets():
    """演示所有三个数据集的超图可视化"""
    print("🔍 Demo: All Datasets Hypergraph Visualization")
    print("="*60)
    
    visualize_multiple_datasets()


def interactive_demo():
    """交互式演示模式"""
    print("🎮 Interactive Hypergraph Visualization Demo")
    print("="*50)
    print("Please select a demo mode:")
    print("1. Single dataset (PathMNIST only)")
    print("2. Two datasets comparison (PathMNIST + BloodMNIST)")
    print("3. All three datasets (PathMNIST + BloodMNIST + Synthetic)")
    print("4. Custom dataset selection")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                demo_single_dataset()
                break
            elif choice == '2':
                demo_custom_datasets(['pathmnist', 'bloodmnist'], 256)
                break
            elif choice == '3':
                demo_all_datasets()
                break
            elif choice == '4':
                print("\nAvailable datasets: pathmnist, bloodmnist, synthetic")
                datasets_input = input("Enter dataset names (comma-separated): ").strip()
                datasets = [d.strip() for d in datasets_input.split(',') if d.strip()]
                
                if datasets:
                    samples_input = input("Enter number of samples per dataset (default 256): ").strip()
                    try:
                        num_samples = int(samples_input) if samples_input else 256
                    except ValueError:
                        num_samples = 256
                    
                    demo_custom_datasets(datasets, num_samples)
                else:
                    print("❌ No valid datasets specified.")
                break
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset Hypergraph Visualization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_multi_hypergraph.py                    # Run all datasets demo
  python demo_multi_hypergraph.py --single           # Run single dataset demo  
  python demo_multi_hypergraph.py --interactive      # Interactive mode
  python demo_multi_hypergraph.py --datasets pathmnist bloodmnist --samples 128
        """
    )
    
    parser.add_argument(
        '--single', 
        action='store_true',
        help='Run single dataset demo (PathMNIST only)'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['pathmnist', 'bloodmnist', 'synthetic'],
        help='Specify datasets to use (pathmnist, bloodmnist, synthetic)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=256,
        help='Number of samples per dataset (default: 256)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_demo()
        elif args.single:
            demo_single_dataset()
        elif len(args.datasets) == 1:
            demo_custom_datasets(args.datasets, args.samples)
        else:
            demo_custom_datasets(args.datasets, args.samples)
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
