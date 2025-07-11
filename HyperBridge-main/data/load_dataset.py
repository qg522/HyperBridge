from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO, Evaluator
from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def load_medmnist(batch_size=64, download=True):
    """
    加载多个MedMNIST数据集
    Returns:
        data_loaders: 数据加载器字典，格式为 {dataset_name: {'train': loader, 'val': loader, 'test': loader}}
    """
    datasets_to_load = ['pathmnist', 'bloodmnist', 'organsmnist']
    data_loaders = {}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    for data_flag in datasets_to_load:
        try:
            info = INFO[data_flag]
            DataClass = getattr(medmnist, info['python_class'])

            train_dataset = DataClass(split='train', transform=transform, download=download)
            val_dataset = DataClass(split='val', transform=transform, download=download)
            test_dataset = DataClass(split='test', transform=transform, download=download)

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

            data_loaders[data_flag] = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            
            print(f"✅ Loaded {data_flag}: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
            
        except Exception as e:
            print(f"⚠️  Failed to load {data_flag}: {e}")
            continue

    return data_loaders


if __name__ == '__main__':
    # 加载数据集
    loaders = load_medmnist()
    for name, splits in loaders.items():
        print(f"Loaded dataset: {name}")
        print(f" - Train batches: {len(splits['train'])}")
        print(f" - Val batches: {len(splits['val'])}")
        print(f" - Test batches: {len(splits['test'])}")
    
    # 导入预处理模块并转换为超图
    try:
        from preprocess import process_multimodal_datasets
        
        print("\n" + "="*60)
        print("Converting datasets to hypergraphs...")
        print("="*60)
        
        # 处理多模态数据集转换为超图
        hypergraph_results = process_multimodal_datasets(loaders, num_samples=256)
        
        print("\n" + "="*60)
        print("Hypergraph conversion completed!")
        print("Results saved in 'hypergraph_results' directory")
        print("="*60)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure preprocess.py is in the same directory")
    except Exception as e:
        print(f"❌ Error during hypergraph processing: {e}")
        import traceback
        traceback.print_exc()


