import torch
import numpy as np
import matplotlib.pyplot as plt
import hypernetx as hnx
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# 添加MedMNIST数据集支持
try:
    from medmnist import INFO, Evaluator
    import medmnist
except ImportError:
    print("MedMNIST not installed. Installing now...")
    import subprocess

    subprocess.run(['pip', 'install', 'medmnist'])
    from medmnist import INFO, Evaluator
    import medmnist

# 添加转换支持
from torchvision import transforms


class HypergraphConverter:
    """使用HyperNetX将图像数据转换为超图数据结构"""

    def __init__(self, similarity_threshold=0.7, max_hyperedge_size=12, feature_dim=64):
        self.similarity_threshold = similarity_threshold
        self.max_hyperedge_size = max_hyperedge_size
        self.feature_dim = feature_dim

    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        从图像中提取特征向量
        Args:
            images: 形状为 (N, C, H, W) 的图像张量
        Returns:
            特征矩阵 (N, feature_dim)
        """
        print("Extracting features from images...")
        batch_size = images.shape[0]

        # 展平图像
        features = images.view(batch_size, -1).numpy()

        # 使用PCA降维
        n_components = min(self.feature_dim, features.shape[1], features.shape[0])
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features)

        print(f"Features extracted: {features_reduced.shape}")
        return features_reduced

    def compute_similarity_matrix(self, features: np.ndarray, metric='cosine') -> np.ndarray:
        """计算特征间的相似度矩阵"""
        print("Computing similarity matrix...")
        if metric == 'cosine':
            similarity_matrix = cosine_similarity(features)
        elif metric == 'euclidean':
            distance_matrix = euclidean_distances(features)
            # 转换为相似度 (距离越小，相似度越大)
            similarity_matrix = 1 / (1 + distance_matrix)
        else:
            raise ValueError("Metric must be 'cosine' or 'euclidean'")

        return similarity_matrix

    def generate_hyperedges_dict(self, similarity_matrix: np.ndarray, labels: np.ndarray = None) -> Dict:
        """
        基于相似度矩阵生成超边字典，适用于HyperNetX
        Returns:
            超边字典，格式为 {edge_name: [node_list]}
        """
        print("Generating hyperedges...")
        n_nodes = similarity_matrix.shape[0]
        hyperedges_dict = {}
        edge_counter = 0

        # 方法1: 基于相似度阈值的超边生成
        for i in range(n_nodes):
            similar_nodes = np.where(similarity_matrix[i] > self.similarity_threshold)[0]

            if len(similar_nodes) > 1:
                # 限制超边大小
                if len(similar_nodes) > self.max_hyperedge_size:
                    similarities = similarity_matrix[i][similar_nodes]
                    top_indices = np.argsort(similarities)[-self.max_hyperedge_size:]
                    similar_nodes = similar_nodes[top_indices]

                edge_name = f"similarity_edge_{edge_counter}"
                hyperedges_dict[edge_name] = [f"node_{j}" for j in similar_nodes]
                edge_counter += 1

        # 方法2: 基于标签的超边生成
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                label_nodes = np.where(labels == label)[0]
                if len(label_nodes) > 1:
                    edge_name = f"label_edge_{label}"
                    hyperedges_dict[edge_name] = [f"node_{j}" for j in label_nodes]

        # 去除重复的超边
        unique_hyperedges = {}
        seen_edges = set()
        for edge_name, nodes in hyperedges_dict.items():
            nodes_tuple = tuple(sorted(nodes))
            if nodes_tuple not in seen_edges:
                seen_edges.add(nodes_tuple)
                unique_hyperedges[edge_name] = nodes

        print(f"Generated {len(unique_hyperedges)} unique hyperedges")
        return unique_hyperedges

    def set_node_attributes_safe(self, hypergraph, attributes_dict):
        """
        安全地设置节点属性，兼容不同版本的HyperNetX
        """
        try:
            # 尝试使用新版本的方法
            if hasattr(hnx, 'set_node_attributes'):
                hnx.set_node_attributes(hypergraph, attributes_dict)
            # 尝试使用旧版本的方法
            elif hasattr(hypergraph, 'set_node_attributes'):
                hypergraph.set_node_attributes(attributes_dict)
            # 如果都没有，直接设置属性
            else:
                # 直接修改hypergraph的内部属性
                if not hasattr(hypergraph, '_node_attrs'):
                    hypergraph._node_attrs = {}
                hypergraph._node_attrs.update(attributes_dict)
                print("Set node attributes using direct assignment")
        except Exception as e:
            print(f"Warning: Could not set node attributes: {e}")
            print("Continuing without node attributes...")

    def convert_to_hypergraph(self, images: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, Any]:
        """
        将图像数据转换为HyperNetX超图表示
        """
        print(f"\n{'=' * 50}")
        print(f"Converting {len(images)} images to hypergraph...")
        print(f"{'=' * 50}")

        # 提取特征
        features = self.extract_features(images)

        # 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(features)

        # 生成超边
        labels_np = labels.numpy().flatten() if labels is not None else None
        hyperedges_dict = self.generate_hyperedges_dict(similarity_matrix, labels_np)

        # 创建HyperNetX超图
        hypergraph = hnx.Hypergraph(hyperedges_dict)

        # 创建节点属性字典
        node_attrs = {}
        for i in range(len(images)):
            node_name = f"node_{i}"
            node_attrs[node_name] = {
                'features': features[i],
                'label': int(labels_np[i]) if labels_np is not None else None,
                'index': i
            }

        # 修复：兼容不同版本的HyperNetX - 只对存在的节点设置属性
        attributes_dict = {}
        for node in hypergraph.nodes:
            if node in node_attrs:
                attributes_dict[node] = node_attrs[node]

        # 使用安全的方法设置节点属性
        self.set_node_attributes_safe(hypergraph, attributes_dict)

        hypergraph_data = {
            'hypergraph': hypergraph,
            'features': features,
            'similarity_matrix': similarity_matrix,
            'hyperedges_dict': hyperedges_dict,
            'node_attrs': attributes_dict,
            'labels': labels_np,
            'n_nodes': len(hypergraph.nodes),
            'n_hyperedges': len(hypergraph.edges)
        }

        print(
            f"Hypergraph created with {hypergraph_data['n_nodes']} nodes and {hypergraph_data['n_hyperedges']} hyperedges")
        return hypergraph_data


class HypergraphVisualizer:
    """使用HyperNetX的超图数据可视化工具"""

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('default')

    def visualize_similarity_matrix(self, similarity_matrix: np.ndarray, dataset_name: str, save_path: str = None):
        """可视化相似度矩阵"""
        plt.figure(figsize=self.figsize)
        sns.heatmap(similarity_matrix, cmap='viridis', square=True, cbar=True)
        plt.title(f'{dataset_name} - Similarity Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Image Index', fontsize=12)
        plt.ylabel('Image Index', fontsize=12)

        if save_path:
            plt.savefig(f"{save_path}/{dataset_name}_similarity_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_feature_distribution(self, features: np.ndarray, labels: np.ndarray,
                                       dataset_name: str, save_path: str = None):
        """可视化特征分布（使用t-SNE降维）"""
        print(f"Computing t-SNE for {dataset_name}...")

        # 如果样本数量太少，调整perplexity
        n_samples = features.shape[0]
        perplexity = min(30, max(5, n_samples // 4))

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=self.figsize)
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
            plt.legend()
        else:
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=50)

        plt.title(f'{dataset_name} - Feature Distribution (t-SNE)', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(f"{save_path}/{dataset_name}_feature_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_hypergraph_structure(self, hypergraph_data: Dict[str, Any],
                                       dataset_name: str, save_path: str = None):
        """可视化超图结构"""
        hypergraph = hypergraph_data['hypergraph']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 左图：节点-超边二分图
        try:
            # 尝试使用不同的布局方法
            if hasattr(hnx.drawing, 'layout') and hasattr(hnx.drawing.layout, 'random_layout'):
                pos = hnx.drawing.layout.random_layout(hypergraph)
            else:
                # 如果没有random_layout，使用spring_layout或其他布局
                pos = None

            hnx.draw(hypergraph, pos=pos, ax=ax1, node_labels=False,
                     node_size=100, edge_size=20)
            ax1.set_title(f'{dataset_name} - Hypergraph Layout\n'
                          f'Nodes: {hypergraph_data["n_nodes"]}, Hyperedges: {hypergraph_data["n_hyperedges"]}',
                          fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"Warning: Could not create hypergraph layout: {e}")
            ax1.text(0.5, 0.5,
                     f'Layout failed\nNodes: {hypergraph_data["n_nodes"]}\nHyperedges: {hypergraph_data["n_hyperedges"]}',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=12)

        # 右图：简化的网络图
        try:
            # 创建节点的2-uniform投影（边图）
            if hasattr(hypergraph, 'dual'):
                G = hypergraph.dual()
            else:
                # 如果没有dual方法，创建一个简单的邻接图
                import networkx as nx
                G = nx.Graph()
                # 添加基于超边的连接
                for edge_name, nodes in hypergraph_data['hyperedges_dict'].items():
                    for i in range(len(nodes)):
                        for j in range(i + 1, len(nodes)):
                            G.add_edge(nodes[i], nodes[j])

            if len(G.nodes) > 0:
                import networkx as nx
                pos = nx.spring_layout(G, k=1, iterations=50)
                nx.draw(G, pos, ax=ax2, node_color='lightblue',
                        node_size=100, alpha=0.8, edge_color='gray', width=0.5)
                ax2.set_title(f'{dataset_name} - Node Connections Graph',
                              fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No edges to display', ha='center', va='center',
                         transform=ax2.transAxes, fontsize=12)
        except Exception as e:
            print(f"Warning: Could not create connection graph: {e}")
            ax2.text(0.5, 0.5, f'Connection graph failed\n{str(e)}', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/{dataset_name}_hypergraph_structure.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_hypergraph_statistics(self, hypergraph_data: Dict[str, Any],
                                        dataset_name: str, save_path: str = None):
        """可视化超图统计信息"""
        hypergraph = hypergraph_data['hypergraph']

        # 计算统计信息
        edge_sizes = [len(hypergraph.edges[edge]) for edge in hypergraph.edges]
        node_degrees = [hypergraph.degree(node) for node in hypergraph.nodes]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 超边大小分布
        if edge_sizes:
            ax1.hist(edge_sizes, bins=range(2, max(edge_sizes) + 2), alpha=0.7, edgecolor='black', color='skyblue')
            ax1.set_xlabel('Hyperedge Size', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title(f'{dataset_name} - Hyperedge Size Distribution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No hyperedges', ha='center', va='center', transform=ax1.transAxes)

        # 节点度分布
        if node_degrees:
            ax2.hist(node_degrees, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
            ax2.set_xlabel('Node Degree', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title(f'{dataset_name} - Node Degree Distribution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No nodes', ha='center', va='center', transform=ax2.transAxes)

        # 超图连通性分析
        try:
            if hasattr(hnx.algorithms, 'connected_components'):
                components = hnx.algorithms.connected_components(hypergraph)
            elif hasattr(hnx, 'connected_components'):
                components = hnx.connected_components(hypergraph)
            else:
                components = []

            component_sizes = [len(comp) for comp in components]
            if component_sizes:
                ax3.bar(range(len(component_sizes)), component_sizes, alpha=0.7, color='lightgreen')
                ax3.set_xlabel('Component Index', fontsize=12)
                ax3.set_ylabel('Component Size', fontsize=12)
                ax3.set_title(f'{dataset_name} - Connected Components', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No components found', ha='center', va='center', transform=ax3.transAxes)
        except Exception as e:
            ax3.text(0.5, 0.5, f'Component analysis not available\n{str(e)}', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=10)

        # 标签分布（如果有）
        labels = hypergraph_data['labels']
        if labels is not None:
            unique_labels, counts = np.unique(labels, return_counts=True)
            ax4.bar(unique_labels, counts, alpha=0.7, color='gold')
            ax4.set_xlabel('Class Label', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title(f'{dataset_name} - Label Distribution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No labels available', ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/{dataset_name}_hypergraph_statistics.png", dpi=300, bbox_inches='tight')
        plt.show()

    def create_comprehensive_visualization(self, hypergraph_data: Dict[str, Any],
                                           dataset_name: str, save_path: str = None):
        """创建综合可视化"""
        print(f"\n{'=' * 50}")
        print(f"Creating comprehensive visualization for {dataset_name}...")
        print(f"{'=' * 50}")

        # 1. 相似度矩阵
        self.visualize_similarity_matrix(hypergraph_data['similarity_matrix'],
                                         dataset_name, save_path)

        # 2. 特征分布
        self.visualize_feature_distribution(hypergraph_data['features'],
                                            hypergraph_data['labels'],
                                            dataset_name, save_path)

        # 3. 超图结构
        self.visualize_hypergraph_structure(hypergraph_data, dataset_name, save_path)

        # 4. 超图统计
        self.visualize_hypergraph_statistics(hypergraph_data, dataset_name, save_path)


def load_pathmnist_dataset(batch_size=128, num_samples=512):
    """
    加载PathMNIST数据集并正确转换为张量
    Args:
        batch_size: 批处理大小
        num_samples: 用于构建超图的样本数量
    Returns:
        图像和标签张量
    """
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # 定义转换：将PIL图像转换为张量
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载数据集并应用转换
    train_dataset = DataClass(split='train', download=True, transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 获取足够的样本
    images_list, labels_list = [], []
    total_samples = 0
    data_iter = iter(train_loader)

    while total_samples < num_samples:
        try:
            # 确保数据是张量
            data, target = next(data_iter)

            # 检查数据类型并转换为张量
            if isinstance(data, list):
                data = torch.stack(data)
            if isinstance(target, list):
                target = torch.stack(target)

            images_list.append(data)
            labels_list.append(target)
            total_samples += data.shape[0]
        except StopIteration:
            # 如果数据用完了，重新开始迭代
            data_iter = iter(train_loader)

    images = torch.cat(images_list)[:num_samples]
    labels = torch.cat(labels_list)[:num_samples].squeeze()

    print(f"Loaded PathMNIST dataset with {images.shape[0]} samples")
    print(f"Image shape: {images.shape[1:]}")
    print(f"Number of classes: {len(torch.unique(labels))}")

    return images, labels


def visualize_hypergraph_for_pathmnist():
    """
    为PathMNIST数据集创建超图并可视化
    """
    # 设置保存目录
    save_dir = "pathmnist_hypergraph_results"
    os.makedirs(save_dir, exist_ok=True)

    # 初始化转换器和可视化器
    converter = HypergraphConverter(similarity_threshold=0.7, max_hyperedge_size=12, feature_dim=64)
    visualizer = HypergraphVisualizer()

    print(f"\n{'=' * 60}")
    print("Processing PathMNIST dataset")
    print(f"{'=' * 60}")

    try:
        # 加载PathMNIST数据集
        images, labels = load_pathmnist_dataset(num_samples=512)

        # 确保图像是浮点张量
        if images.dtype != torch.float32:
            images = images.float()

        # 转换为超图
        hypergraph_data = converter.convert_to_hypergraph(images, labels)

        # 保存超图数据
        save_file = f"{save_dir}/pathmnist_hypergraph.pkl"
        with open(save_file, 'wb') as f:
            # 保存时排除HyperNetX对象
            save_data = hypergraph_data.copy()
            save_data['hypergraph_edges'] = hypergraph_data['hyperedges_dict']
            del save_data['hypergraph']  # 临时移除HyperNetX对象
            pickle.dump(save_data, f)
        print(f"Hypergraph data saved to: {save_file}")

        # 创建可视化
        visualizer.create_comprehensive_visualization(hypergraph_data,
                                                      "Pathology PathMNIST",
                                                      save_dir)

        # 打印详细统计信息
        hypergraph = hypergraph_data['hypergraph']
        print(f"\n📊 PATHMNIST STATISTICS:")
        print(f"  🔹 Nodes: {hypergraph_data['n_nodes']}")
        print(f"  🔹 Hyperedges: {hypergraph_data['n_hyperedges']}")
        if hypergraph_data['n_hyperedges'] > 0:
            edge_sizes = [len(hypergraph.edges[edge]) for edge in hypergraph.edges]
            print(f"  🔹 Average hyperedge size: {np.mean(edge_sizes):.2f}")
            print(f"  🔹 Max hyperedge size: {max(edge_sizes)}")
            print(f"  🔹 Min hyperedge size: {min(edge_sizes)}")
        print(f"  🔹 Feature dimension: {hypergraph_data['features'].shape[1]}")

        # 连通性分析
        try:
            if hasattr(hnx.algorithms, 'connected_components'):
                components = list(hnx.algorithms.connected_components(hypergraph))
            elif hasattr(hnx, 'connected_components'):
                components = list(hnx.connected_components(hypergraph))
            else:
                components = []

            print(f"  🔹 Connected components: {len(components)}")
            if components:
                largest_component_size = max(len(comp) for comp in components)
                print(f"  🔹 Largest component size: {largest_component_size}")
        except Exception as e:
            print(f"  🔹 Connected components: Analysis not available ({e})")

    except Exception as e:
        print(f"❌ Error processing PathMNIST: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("🎉 HYPERGRAPH CONVERSION COMPLETED!")
    print(f"📁 Results saved in '{save_dir}' directory")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # 处理PathMNIST数据集
    visualize_hypergraph_for_pathmnist()