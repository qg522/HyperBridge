#!/usr/bin/env python3
"""
测试数据加载和超图转换功能
Test data loading and hypergraph conversion functionality
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """测试数据加载功能"""
    print("🧪 Testing Data Loading Functionality")
    print("="*50)
    
    try:
        from load_dataset import load_medmnist
        
        print("📊 Loading MedMNIST datasets...")
        loaders = load_medmnist(batch_size=32, download=True)
        
        print(f"\n✅ Successfully loaded {len(loaders)} datasets:")
        for name, splits in loaders.items():
            print(f"  - {name.upper()}:")
            print(f"    Train batches: {len(splits['train'])}")
            print(f"    Val batches: {len(splits['val'])}")
            print(f"    Test batches: {len(splits['test'])}")
        
        return loaders
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_hypergraph_conversion(loaders):
    """测试超图转换功能"""
    print("\n🔄 Testing Hypergraph Conversion")
    print("="*50)
    
    if loaders is None:
        print("❌ No data loaders available for testing")
        return
    
    try:
        from preprocess import process_multimodal_datasets
        
        print("🚀 Converting datasets to hypergraphs...")
        # 使用较小的样本数量进行快速测试
        hypergraph_results = process_multimodal_datasets(loaders, num_samples=64, save_dir="test_hypergraph_results")
        
        print(f"\n✅ Successfully converted {len(hypergraph_results)} datasets to hypergraphs:")
        for dataset_name, data in hypergraph_results.items():
            print(f"  - {dataset_name.upper()}:")
            print(f"    Nodes: {data['n_nodes']}")
            print(f"    Hyperedges: {data['n_hyperedges']}")
            print(f"    Feature dim: {data['features'].shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hypergraph conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocess_functions():
    """测试预处理模块的各个函数"""
    print("\n🔧 Testing Individual Preprocess Functions")
    print("="*50)
    
    try:
        from preprocess import HypergraphConverter, HypergraphVisualizer, load_medmnist_dataset
        
        # 测试HypergraphConverter
        print("1. Testing HypergraphConverter...")
        converter = HypergraphConverter(similarity_threshold=0.7, max_hyperedge_size=12, feature_dim=32)
        print("   ✅ HypergraphConverter created successfully")
        
        # 测试HypergraphVisualizer
        print("2. Testing HypergraphVisualizer...")
        visualizer = HypergraphVisualizer(figsize=(10, 6))
        print("   ✅ HypergraphVisualizer created successfully")
        
        # 测试单个数据集加载
        print("3. Testing single dataset loading...")
        images, labels = load_medmnist_dataset('pathmnist', num_samples=32)
        print(f"   ✅ Loaded PathMNIST: {images.shape}, Labels: {labels.shape}")
        
        # 测试超图转换
        print("4. Testing hypergraph conversion...")
        hypergraph_data = converter.convert_to_hypergraph(images, labels)
        print(f"   ✅ Hypergraph created: {hypergraph_data['n_nodes']} nodes, {hypergraph_data['n_hyperedges']} edges")
        
        return True
        
    except Exception as e:
        print(f"❌ Function testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Data Loading and Hypergraph Conversion Test Suite")
    print("="*60)
    
    # 测试数据加载
    loaders = test_data_loading()
    
    # 测试预处理功能
    preprocess_success = test_preprocess_functions()
    
    # 测试超图转换（如果数据加载成功）
    if loaders and preprocess_success:
        conversion_success = test_hypergraph_conversion(loaders)
        
        if conversion_success:
            print(f"\n🎉 All tests completed successfully!")
            print("Your data loading and hypergraph conversion pipeline is working correctly.")
        else:
            print(f"\n⚠️  Hypergraph conversion test failed, but basic functions work.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the error messages above.")
    
    print(f"\n{'='*60}")
    print("Testing completed!")

if __name__ == "__main__":
    main()
