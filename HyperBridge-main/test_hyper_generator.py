#!/usr/bin/env python3
"""
Test HybridHyperedgeGenerator module with MedMNIST data and visualize results
使用MedMNIST数据测试HybridHyperedgeGenerator模块并可视化
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_dataset import load_medmnist
from models.modules.hyper_generator import HybridHyperedgeGenerator


class HypergraphTester:
    """Hypergraph generator testing and visualization class"""
    
    def __init__(self, hidden_dim=64, top_k=8, threshold=0.5):
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.threshold = threshold
        
        # Create simple feature extractors to simulate multimodal encoders
        # Fix: Use adaptive pooling and dynamic linear layer
        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # Pool images to 8x8
            nn.Flatten(),
            # For 3-channel images, pooled dimension is 3*8*8=192
            nn.Linear(192, hidden_dim),  # Fix: 192 = 3 channels * 8 * 8
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Simulate text encoder (using random vectors as replacement)
        self.text_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Simulate signal encoder
        self.signal_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Hypergraph generator
        self.hypergraph_generator = HybridHyperedgeGenerator(
            num_modalities=3,
            input_dims=[hidden_dim, hidden_dim, hidden_dim],
            hidden_dim=hidden_dim,
            top_k=top_k,
            threshold=threshold
        )
    
    def extract_features_from_batch(self, images, labels, batch_size=32):
        """Extract multimodal features from a batch of MedMNIST data"""
        print(f"Extracting features... Input shape: {images.shape}")
        
        # Handle image dimensions
        actual_batch_size = images.shape[0]
        
        # Image features
        if len(images.shape) == 3:  # Grayscale [B, H, W]
            images = images.unsqueeze(1)  # Add channel dimension [B, 1, H, W]
        elif len(images.shape) == 4 and images.shape[1] == 1:  # Grayscale [B, 1, H, W]
            pass
        elif len(images.shape) == 4 and images.shape[1] == 3:  # RGB [B, 3, H, W]
            pass
        else:
            # Other formats, convert to standard format
            images = images.view(actual_batch_size, 1, 28, 28)
        
        # Ensure 3-channel images (duplicate grayscale channels)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        print(f"Processed image shape: {images.shape}")
        
        with torch.no_grad():
            # Image features
            img_features = self.image_encoder(images)  # [B, hidden_dim]
            
            # Simulate text features (generate meaningful features based on labels)
            text_features = torch.zeros(actual_batch_size, self.hidden_dim)
            for i, label in enumerate(labels):
                # Generate different text feature patterns for different classes
                base_pattern = torch.randn(self.hidden_dim) * 0.1
                class_pattern = torch.zeros(self.hidden_dim)
                class_pattern[label.item() * 8:(label.item() + 1) * 8] = 1.0  # Class-specific pattern
                text_features[i] = base_pattern + class_pattern * 0.5
            
            text_features = self.text_encoder(text_features)
            
            # Simulate signal features (based on image statistics)
            signal_features = torch.zeros(actual_batch_size, self.hidden_dim)
            for i in range(actual_batch_size):
                # Statistical features based on images
                img = images[i]
                mean_val = img.mean()
                std_val = img.std()
                max_val = img.max()
                
                # Create signal features
                signal_pattern = torch.randn(self.hidden_dim) * 0.1
                signal_pattern[:16] = mean_val
                signal_pattern[16:32] = std_val
                signal_pattern[32:48] = max_val
                signal_pattern[48:] = labels[i].float() * 0.1
                
                signal_features[i] = signal_pattern
            
            signal_features = self.signal_encoder(signal_features)
        
        return img_features, text_features, signal_features
    
    def test_hypergraph_generation(self, dataset_name='pathmnist', num_samples=64):
        """Test hypergraph generation"""
        print(f"\n{'='*60}")
        print(f"Testing Hypergraph Generation - Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load data
        print("Loading data...")
        data_loaders = load_medmnist(batch_size=32)
        
        if dataset_name not in data_loaders:
            print(f"❌ Dataset {dataset_name} not found")
            return None
        
        train_loader = data_loaders[dataset_name]['train']
        
        # Collect data
        all_images = []
        all_labels = []
        collected_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            all_images.append(images)
            all_labels.append(labels.squeeze())
            collected_samples += images.shape[0]
            
            if collected_samples >= num_samples:
                break
        
        # Merge data
        all_images = torch.cat(all_images)[:num_samples]
        all_labels = torch.cat(all_labels)[:num_samples]
        
        print(f"Collected data: {all_images.shape}, Labels: {all_labels.shape}")
        print(f"Unique labels: {torch.unique(all_labels).tolist()}")
        
        # Extract multimodal features
        img_features, text_features, signal_features = self.extract_features_from_batch(
            all_images, all_labels, num_samples
        )
        
        print(f"Feature shapes:")
        print(f"  Image features: {img_features.shape}")
        print(f"  Text features: {text_features.shape}")
        print(f"  Signal features: {signal_features.shape}")
        
        # Generate hypergraph
        print("Generating hypergraph...")
        try:
            H, edge_weights = self.hypergraph_generator([img_features, text_features, signal_features])
            
            print(f"✅ Hypergraph generation successful:")
            print(f"  Incidence matrix shape: {H.shape}")
            print(f"  Active hyperedges count: {len(edge_weights)}")
            print(f"  Average hyperedge weight: {edge_weights.mean().item():.4f}")
            
            return {
                'H': H.cpu().numpy(),
                'edge_weights': edge_weights.cpu().numpy(),
                'img_features': img_features.cpu().numpy(),
                'text_features': text_features.cpu().numpy(),
                'signal_features': signal_features.cpu().numpy(),
                'labels': all_labels.cpu().numpy(),
                'images': all_images.cpu().numpy(),
                'dataset_name': dataset_name
            }
            
        except Exception as e:
            print(f"❌ Hypergraph generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_hypergraph_results(self, results, save_dir='hypergraph_test_results'):
        """Visualize hypergraph results"""
        if results is None:
            print("❌ No results to visualize")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = results['dataset_name']
        
        print(f"\n📊 Creating visualization charts...")
        
        # 1. Hypergraph structure statistics
        self._plot_hypergraph_statistics(results, save_dir)
        
        # 2. Feature space visualization
        self._plot_feature_spaces(results, save_dir)
        
        # 3. Hypergraph connectivity analysis
        self._plot_connectivity_analysis(results, save_dir)
        
        # 4. Hyperedge weight analysis
        self._plot_edge_weight_analysis(results, save_dir)
        
        print(f"✅ Visualization completed, results saved in '{save_dir}' directory")
    
    def _plot_hypergraph_statistics(self, results, save_dir):
        """Plot hypergraph statistics"""
        H = results['H']
        edge_weights = results['edge_weights']
        labels = results['labels']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{results["dataset_name"].upper()} - Hypergraph Statistical Analysis', fontsize=16, fontweight='bold')
        
        # Node degree distribution
        node_degrees = H.sum(axis=1)
        axes[0, 0].hist(node_degrees, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Node Degree Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Node Degree')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hyperedge size distribution
        edge_sizes = H.sum(axis=0)
        if len(edge_sizes) > 0:
            axes[0, 1].hist(edge_sizes, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_title('Hyperedge Size Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Hyperedge Size')
            axes[0, 1].set_ylabel('Frequency')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Hyperedges', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hyperedge weight distribution
        if len(edge_weights) > 0:
            axes[1, 0].hist(edge_weights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_title('Hyperedge Weight Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Weight')
            axes[1, 0].set_ylabel('Frequency')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Weight Data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        axes[1, 1].bar(unique_labels, counts, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('Class Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hypergraph_statistics.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_spaces(self, results, save_dir):
        """Plot feature spaces"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{results["dataset_name"].upper()} - Multimodal Feature Spaces', fontsize=16, fontweight='bold')
        
        labels = results['labels']
        colors = plt.cm.tab10(labels)
        
        features_list = [
            ('Image Features', results['img_features']),
            ('Text Features', results['text_features']),
            ('Signal Features', results['signal_features']),
        ]
        
        for idx, (title, features) in enumerate(features_list):
            if idx >= 3:
                break
            
            ax = axes[idx // 2, idx % 2]
            
            try:
                # t-SNE dimensionality reduction
                if features.shape[1] > 2:
                    perplexity = min(30, max(5, features.shape[0] // 4))
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    features_2d = tsne.fit_transform(features)
                else:
                    features_2d = features
                
                # Plot scatter
                scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                                   c=colors, alpha=0.7, s=50)
                
                # Add legend
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                             label=f'Class {label}', alpha=0.7, s=50)
                
                ax.set_title(f'{title} (t-SNE)', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Visualization failed:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title} (Error)', fontweight='bold')
        
        # Incidence matrix heatmap
        ax = axes[1, 1]
        H = results['H']
        im = ax.imshow(H, cmap='Blues', aspect='auto')
        ax.set_title('Hypergraph Incidence Matrix', fontweight='bold')
        ax.set_xlabel('Hyperedges')
        ax.set_ylabel('Nodes')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_spaces.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_connectivity_analysis(self, results, save_dir):
        """Plot connectivity analysis"""
        H = results['H']
        labels = results['labels']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{results["dataset_name"].upper()} - Connectivity Analysis', fontsize=16, fontweight='bold')
        
        # Create connection graph based on hypergraph
        G = nx.Graph()
        n_nodes = H.shape[0]
        G.add_nodes_from(range(n_nodes))
        
        # Add edges based on hyperedges
        for e_idx in range(H.shape[1]):
            nodes_in_edge = np.where(H[:, e_idx] == 1)[0]
            for i in range(len(nodes_in_edge)):
                for j in range(i + 1, len(nodes_in_edge)):
                    node_i, node_j = nodes_in_edge[i], nodes_in_edge[j]
                    if G.has_edge(node_i, node_j):
                        G[node_i][node_j]['weight'] += 1
                    else:
                        G.add_edge(node_i, node_j, weight=1)
        
        # Plot connection graph
        try:
            pos = nx.spring_layout(G, k=1, iterations=50)
            node_colors = [plt.cm.tab10(labels[i]) for i in range(len(labels))]
            
            if len(G.edges()) > 0:
                edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
                nx.draw(G, pos, ax=axes[0], node_color=node_colors, node_size=50,
                       width=[w * 0.5 for w in edge_weights], alpha=0.8, edge_color='gray')
            else:
                nx.draw(G, pos, ax=axes[0], node_color=node_colors, node_size=50,
                       alpha=0.8)
            
            axes[0].set_title('Node Connection Graph', fontweight='bold')
            
            # Connection degree distribution
            degrees = [G.degree(n) for n in G.nodes()]
            axes[1].hist(degrees, bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[1].set_title('Connection Degree Distribution', fontweight='bold')
            axes[1].set_xlabel('Connection Degree')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[0].text(0.5, 0.5, f'Graph visualization failed:\n{str(e)}', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Node Connection Graph (Error)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/connectivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_edge_weight_analysis(self, results, save_dir):
        """Plot hyperedge weight analysis"""
        edge_weights = results['edge_weights']
        labels = results['labels']
        H = results['H']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{results["dataset_name"].upper()} - Hyperedge Weight Analysis', fontsize=16, fontweight='bold')
        
        if len(edge_weights) > 0:
            # Weight vs hyperedge size
            edge_sizes = H.sum(axis=0)
            axes[0, 0].scatter(edge_sizes, edge_weights, alpha=0.7, color='purple')
            axes[0, 0].set_xlabel('Hyperedge Size')
            axes[0, 0].set_ylabel('Hyperedge Weight')
            axes[0, 0].set_title('Hyperedge Size vs Weight', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Weight cumulative distribution
            sorted_weights = np.sort(edge_weights)
            cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
            axes[0, 1].plot(sorted_weights, cumulative, linewidth=2, color='red')
            axes[0, 1].set_xlabel('Hyperedge Weight')
            axes[0, 1].set_ylabel('Cumulative Probability')
            axes[0, 1].set_title('Weight Cumulative Distribution', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Threshold effect analysis
            thresholds = np.linspace(0.1, 0.9, 9)
            active_edges = []
            for thresh in thresholds:
                active_edges.append(np.sum(edge_weights > thresh))
            
            axes[1, 0].plot(thresholds, active_edges, marker='o', linewidth=2, color='green')
            axes[1, 0].set_xlabel('Threshold')
            axes[1, 0].set_ylabel('Active Hyperedges Count')
            axes[1, 0].set_title('Threshold Effect on Active Hyperedges', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Weight distribution box plot
            axes[1, 1].boxplot(edge_weights, vert=True)
            axes[1, 1].set_ylabel('Hyperedge Weight')
            axes[1, 1].set_title('Weight Distribution Box Plot', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
        else:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No Hyperedge Weight Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/edge_weight_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    print("🚀 HybridHyperedgeGenerator Testing and Visualization")
    print("="*70)
    
    # Initialize tester
    tester = HypergraphTester(hidden_dim=64, top_k=8, threshold=0.4)
    
    # Test different datasets
    datasets_to_test = ['pathmnist', 'bloodmnist']  # Can add more datasets
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Testing dataset: {dataset_name.upper()}")
        
        # Generate hypergraph
        results = tester.test_hypergraph_generation(dataset_name, num_samples=64)
        
        if results is not None:
            # Create visualization
            save_dir = f'hypergraph_test_results/{dataset_name}'
            tester.visualize_hypergraph_results(results, save_dir)
            
            # Print statistics
            print(f"\n📈 {dataset_name.upper()} Statistics:")
            print(f"  Node count: {results['H'].shape[0]}")
            print(f"  Hyperedge count: {results['H'].shape[1]}")
            print(f"  Active hyperedge count: {len(results['edge_weights'])}")
            if len(results['edge_weights']) > 0:
                print(f"  Average hyperedge weight: {results['edge_weights'].mean():.4f}")
                print(f"  Average hyperedge size: {results['H'].sum(axis=0).mean():.2f}")
            print(f"  Average node degree: {results['H'].sum(axis=1).mean():.2f}")
        
        print(f"\n" + "-"*50)
    
    print(f"\n🎉 Testing completed!")
    print(f"📁 Results saved in 'hypergraph_test_results' directory")


if __name__ == "__main__":
    main()
