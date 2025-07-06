#!/usr/bin/env python3
"""
Complete pipeline test: Dataset -> HyperGenerator -> Pruning Regularizer
完整流程测试：数据集 -> 超图生成器 -> 剪枝正则化器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_dataset import load_medmnist
from models.modules.hyper_generator import HybridHyperedgeGenerator
from models.modules.pruning_regularizer import SpectralCutRegularizer


class FullPipelineTester:
    """Complete pipeline tester for Dataset -> HyperGenerator -> Pruning"""
    
    def __init__(self, hidden_dim=64, top_k=8, threshold=0.4):
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.threshold = threshold
        
        # Feature extractors for multimodal simulation
        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(192, hidden_dim),  # 3*8*8=192 for RGB images
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.text_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.signal_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Hypergraph generator
        self.hypergraph_generator = HybridHyperedgeGenerator(
            num_modalities=3,
            input_dims=[hidden_dim, hidden_dim, hidden_dim],
            hidden_dim=hidden_dim,
            top_k=top_k,
            threshold=threshold
        )
        
        # Pruning regularizer
        self.pruning_regularizer = SpectralCutRegularizer(
            use_rayleigh=True,
            reduction='mean'
        )
        
        # Node embedding network (for pruning regularizer)
        self.node_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # Concatenate all modality features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def extract_multimodal_features(self, images, labels):
        """Extract multimodal features from images and labels"""
        batch_size = images.shape[0]
        
        # Process image dimensions
        if len(images.shape) == 3:  # Grayscale [B, H, W]
            images = images.unsqueeze(1)  # Add channel dimension [B, 1, H, W]
        
        # Convert to RGB if grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            # Image features
            img_features = self.image_encoder(images)
            
            # Simulate text features based on labels
            text_features = torch.zeros(batch_size, self.hidden_dim)
            for i, label in enumerate(labels):
                base_pattern = torch.randn(self.hidden_dim) * 0.1
                class_pattern = torch.zeros(self.hidden_dim)
                label_idx = min(label.item(), 7)  # Ensure within bounds
                class_pattern[label_idx * 8:(label_idx + 1) * 8] = 1.0
                text_features[i] = base_pattern + class_pattern * 0.5
            
            text_features = self.text_encoder(text_features)
            
            # Simulate signal features based on image statistics
            signal_features = torch.zeros(batch_size, self.hidden_dim)
            for i in range(batch_size):
                img = images[i]
                mean_val = img.mean()
                std_val = img.std()
                max_val = img.max()
                
                signal_pattern = torch.randn(self.hidden_dim) * 0.1
                signal_pattern[:16] = mean_val
                signal_pattern[16:32] = std_val
                signal_pattern[32:48] = max_val
                signal_pattern[48:] = labels[i].float() * 0.1
                
                signal_features[i] = signal_pattern
            
            signal_features = self.signal_encoder(signal_features)
        
        return img_features, text_features, signal_features
    
    def compute_degree_matrices(self, H):
        """Compute node degree matrix Dv and hyperedge degree matrix De"""
        # Node degrees (number of hyperedges each node belongs to)
        node_degrees = H.sum(dim=1)  # [N]
        Dv = torch.diag(node_degrees + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Hyperedge degrees (number of nodes in each hyperedge)
        edge_degrees = H.sum(dim=0)  # [E]
        De = torch.diag(edge_degrees + 1e-8)
        
        return Dv, De
    
    def test_full_pipeline(self, dataset_name='pathmnist', num_samples=64):
        """Test the complete pipeline"""
        print(f"\n{'='*80}")
        print(f"TESTING FULL PIPELINE - Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Step 1: Load dataset
        print("Step 1: Loading dataset...")
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
        
        all_images = torch.cat(all_images)[:num_samples]
        all_labels = torch.cat(all_labels)[:num_samples]
        
        print(f"✅ Loaded {all_images.shape[0]} samples")
        print(f"   Image shape: {all_images.shape[1:]}")
        print(f"   Unique labels: {torch.unique(all_labels).tolist()}")
        
        # Step 2: Extract multimodal features
        print("\nStep 2: Extracting multimodal features...")
        img_features, text_features, signal_features = self.extract_multimodal_features(
            all_images, all_labels
        )
        
        print(f"✅ Feature extraction completed")
        print(f"   Image features: {img_features.shape}")
        print(f"   Text features: {text_features.shape}")
        print(f"   Signal features: {signal_features.shape}")
        
        # Step 3: Generate hypergraph
        print("\nStep 3: Generating hypergraph...")
        try:
            H, edge_weights = self.hypergraph_generator([img_features, text_features, signal_features])
            
            print(f"✅ Hypergraph generation successful")
            print(f"   Incidence matrix shape: {H.shape}")
            print(f"   Active hyperedges: {len(edge_weights)}")
            print(f"   Average edge weight: {edge_weights.mean().item():.4f}")
            
            # Filter out zero columns (inactive hyperedges)
            active_mask = H.sum(dim=0) > 0
            H_active = H[:, active_mask]
            active_edge_weights = edge_weights[:len(active_mask)][active_mask[:len(edge_weights)]]
            
            print(f"   Active hypergraph matrix: {H_active.shape}")
            
        except Exception as e:
            print(f"❌ Hypergraph generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Step 4: Compute degree matrices
        print("\nStep 4: Computing degree matrices...")
        Dv, De = self.compute_degree_matrices(H_active)
        
        print(f"✅ Degree matrices computed")
        print(f"   Node degree matrix Dv: {Dv.shape}")
        print(f"   Edge degree matrix De: {De.shape}")
        print(f"   Average node degree: {torch.diag(Dv).mean().item():.2f}")
        print(f"   Average edge degree: {torch.diag(De).mean().item():.2f}")
        
        # Step 5: Create node embeddings
        print("\nStep 5: Creating node embeddings...")
        # Concatenate all modality features for node embeddings
        combined_features = torch.cat([img_features, text_features, signal_features], dim=1)
        node_embeddings = self.node_embedding(combined_features)
        
        print(f"✅ Node embeddings created: {node_embeddings.shape}")
        
        # Step 6: Apply pruning regularizer
        print("\nStep 6: Applying pruning regularizer...")
        try:
            if H_active.shape[1] > 0:  # Check if there are active hyperedges
                pruning_loss = self.pruning_regularizer(node_embeddings, H_active, Dv, De)
                
                print(f"✅ Pruning regularizer applied successfully")
                print(f"   Spectral cut loss: {pruning_loss.item():.6f}")
            else:
                print("⚠️  No active hyperedges for pruning regularizer")
                pruning_loss = torch.tensor(0.0)
                
        except Exception as e:
            print(f"❌ Pruning regularizer failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Compile results
        results = {
            'dataset_name': dataset_name,
            'images': all_images.cpu().numpy(),
            'labels': all_labels.cpu().numpy(),
            'img_features': img_features.cpu().numpy(),
            'text_features': text_features.cpu().numpy(),
            'signal_features': signal_features.cpu().numpy(),
            'H': H_active.cpu().numpy(),
            'edge_weights': active_edge_weights.cpu().numpy(),
            'node_embeddings': node_embeddings.detach().cpu().numpy(),  # Fix: use detach()
            'Dv': Dv.cpu().numpy(),
            'De': De.cpu().numpy(),
            'pruning_loss': pruning_loss.item(),
            'num_nodes': H_active.shape[0],
            'num_edges': H_active.shape[1],
            'avg_node_degree': torch.diag(Dv).mean().item(),
            'avg_edge_degree': torch.diag(De).mean().item()
        }
        
        print(f"\n{'='*80}")
        print("🎉 FULL PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print(f"📊 Summary Statistics:")
        print(f"   Dataset: {dataset_name.upper()}")
        print(f"   Samples: {results['num_nodes']}")
        print(f"   Active hyperedges: {results['num_edges']}")
        print(f"   Spectral cut loss: {results['pruning_loss']:.6f}")
        print(f"   Average node degree: {results['avg_node_degree']:.2f}")
        print(f"   Average edge degree: {results['avg_edge_degree']:.2f}")
        print(f"{'='*80}")
        
        return results
    
    def visualize_pipeline_results(self, results, save_dir='pipeline_test_results'):
        """Visualize the complete pipeline results"""
        if results is None:
            print("❌ No results to visualize")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = results['dataset_name']
        
        print(f"\n📊 Creating pipeline visualization...")
        
        # Create comprehensive visualization
        self._create_pipeline_visualization(results, save_dir)
        
        print(f"✅ Visualization completed, saved in '{save_dir}'")
    
    def _create_pipeline_visualization(self, results, save_dir):
        """Create comprehensive pipeline visualization"""
        fig = plt.figure(figsize=(20, 16))
        dataset_name = results['dataset_name']
        
        # Main title
        fig.suptitle(f'{dataset_name.upper()} - Complete Pipeline Analysis', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Dataset overview (2x2 subplot in top-left)
        ax1 = plt.subplot(3, 4, 1)
        labels = results['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax1.bar(unique_labels, counts, color='skyblue', alpha=0.8)
        ax1.set_title('Dataset Label Distribution', fontweight='bold')
        ax1.set_xlabel('Class Label')
        ax1.set_ylabel('Sample Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sample images
        ax2 = plt.subplot(3, 4, 2)
        if len(results['images']) > 0:
            # Show a few sample images
            sample_img = results['images'][0]
            if len(sample_img.shape) == 3:  # RGB
                if sample_img.shape[0] == 3:  # Channel first
                    sample_img = sample_img.transpose(1, 2, 0)
                ax2.imshow(sample_img)
            else:  # Grayscale
                ax2.imshow(sample_img, cmap='gray')
            ax2.set_title('Sample Image', fontweight='bold')
            ax2.axis('off')
        
        # 3. Feature space dimensions
        ax3 = plt.subplot(3, 4, 3)
        feature_dims = [
            results['img_features'].shape[1],
            results['text_features'].shape[1], 
            results['signal_features'].shape[1]
        ]
        feature_names = ['Image', 'Text', 'Signal']
        bars = ax3.bar(feature_names, feature_dims, color=['red', 'green', 'blue'], alpha=0.7)
        ax3.set_title('Feature Dimensions', fontweight='bold')
        ax3.set_ylabel('Dimension Size')
        for bar, dim in zip(bars, feature_dims):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(dim), ha='center', va='bottom', fontweight='bold')
        
        # 4. Hypergraph statistics
        ax4 = plt.subplot(3, 4, 4)
        stats = [
            results['num_nodes'],
            results['num_edges'],
            results['avg_node_degree'],
            results['avg_edge_degree']
        ]
        stat_names = ['Nodes', 'Edges', 'Avg Node\nDegree', 'Avg Edge\nDegree']
        colors = ['purple', 'orange', 'pink', 'cyan']
        bars = ax4.bar(stat_names, stats, color=colors, alpha=0.7)
        ax4.set_title('Hypergraph Statistics', fontweight='bold')
        ax4.set_ylabel('Count/Value')
        for bar, stat in zip(bars, stats):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{stat:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Hypergraph incidence matrix heatmap
        ax5 = plt.subplot(3, 4, (5, 6))
        H = results['H']
        if H.shape[1] > 0:
            # Sample a subset if too large
            max_display = 50
            if H.shape[0] > max_display:
                indices = np.random.choice(H.shape[0], max_display, replace=False)
                H_display = H[indices]
            else:
                H_display = H
            
            if H.shape[1] > max_display:
                edge_indices = np.random.choice(H.shape[1], max_display, replace=False)
                H_display = H_display[:, edge_indices]
            
            im = ax5.imshow(H_display, cmap='Blues', aspect='auto')
            ax5.set_title(f'Hypergraph Incidence Matrix ({H_display.shape[0]}x{H_display.shape[1]})', 
                         fontweight='bold')
            ax5.set_xlabel('Hyperedges')
            ax5.set_ylabel('Nodes')
            plt.colorbar(im, ax=ax5, shrink=0.8)
        else:
            ax5.text(0.5, 0.5, 'No Active Hyperedges', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Hypergraph Incidence Matrix', fontweight='bold')
        
        # 6. Node degree distribution
        ax6 = plt.subplot(3, 4, 7)
        if H.shape[1] > 0:
            node_degrees = H.sum(axis=1)
            ax6.hist(node_degrees, bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
            ax6.set_title('Node Degree Distribution', fontweight='bold')
            ax6.set_xlabel('Node Degree')
            ax6.set_ylabel('Frequency')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No Degree Data', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Node Degree Distribution', fontweight='bold')
        
        # 7. Edge weight distribution
        ax7 = plt.subplot(3, 4, 8)
        edge_weights = results['edge_weights']
        if len(edge_weights) > 0:
            ax7.hist(edge_weights, bins=15, color='lightgreen', alpha=0.7, edgecolor='black')
            ax7.set_title('Edge Weight Distribution', fontweight='bold')
            ax7.set_xlabel('Edge Weight')
            ax7.set_ylabel('Frequency')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No Edge Weights', ha='center', va='center',
                    transform=ax7.transAxes, fontsize=14)
            ax7.set_title('Edge Weight Distribution', fontweight='bold')
        
        # 8. Node embedding visualization (2D projection)
        ax8 = plt.subplot(3, 4, (9, 10))
        node_embeddings = results['node_embeddings']
        if node_embeddings.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(node_embeddings)
        else:
            embeddings_2d = node_embeddings
        
        scatter = ax8.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=labels, cmap='tab10', alpha=0.7, s=50)
        ax8.set_title('Node Embeddings (PCA Projection)', fontweight='bold')
        ax8.set_xlabel('PC1')
        ax8.set_ylabel('PC2')
        ax8.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax8, shrink=0.8, label='Class Label')
        
        # 9. Pruning loss and statistics
        ax9 = plt.subplot(3, 4, 11)
        pruning_info = [
            results['pruning_loss'],
            results['avg_node_degree'],
            results['avg_edge_degree'],
            len(edge_weights) / results['num_nodes'] if results['num_nodes'] > 0 else 0
        ]
        pruning_labels = ['Pruning\nLoss', 'Avg Node\nDegree', 'Avg Edge\nDegree', 'Edge\nDensity']
        bars = ax9.bar(pruning_labels, pruning_info, color='gold', alpha=0.7)
        ax9.set_title('Pruning & Graph Metrics', fontweight='bold')
        ax9.set_ylabel('Value')
        for bar, value in zip(bars, pruning_info):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 10. Pipeline summary
        ax10 = plt.subplot(3, 4, 12)
        ax10.axis('off')
        summary_text = f"""
PIPELINE SUMMARY
Dataset: {dataset_name.upper()}
Samples: {results['num_nodes']}
Features: 3 modalities
Hyperedges: {results['num_edges']}
Pruning Loss: {results['pruning_loss']:.4f}

FLOW:
Data → Features → Hypergraph → Pruning
✅ All stages completed successfully
        """
        ax10.text(0.1, 0.9, summary_text.strip(), transform=ax10.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_dir}/complete_pipeline_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to test the complete pipeline"""
    print("🚀 COMPLETE PIPELINE TEST: Dataset → HyperGenerator → Pruning")
    print("="*80)
    
    # Initialize tester
    tester = FullPipelineTester(hidden_dim=64, top_k=8, threshold=0.4)
    
    # Test datasets
    datasets_to_test = ['pathmnist', 'bloodmnist']
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Testing pipeline with {dataset_name.upper()}")
        
        # Run full pipeline test
        results = tester.test_full_pipeline(dataset_name, num_samples=64)
        
        if results is not None:
            # Create visualizations
            save_dir = f'pipeline_test_results/{dataset_name}'
            tester.visualize_pipeline_results(results, save_dir)
            
            print(f"\n📈 {dataset_name.upper()} PIPELINE RESULTS:")
            print(f"  ✅ Data loading: SUCCESS")
            print(f"  ✅ Feature extraction: SUCCESS")
            print(f"  ✅ Hypergraph generation: SUCCESS ({results['num_edges']} edges)")
            print(f"  ✅ Degree computation: SUCCESS")
            print(f"  ✅ Node embedding: SUCCESS")
            print(f"  ✅ Pruning regularizer: SUCCESS (loss: {results['pruning_loss']:.6f})")
        else:
            print(f"  ❌ Pipeline test failed for {dataset_name}")
        
        print(f"\n" + "-"*60)
    
    print(f"\n🎉 COMPLETE PIPELINE TESTING FINISHED!")
    print(f"📁 Results saved in 'pipeline_test_results' directory")


if __name__ == "__main__":
    main()
