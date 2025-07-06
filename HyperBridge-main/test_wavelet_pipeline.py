#!/usr/bin/env python3
"""
Complete pipeline test: Dataset -> HyperGenerator -> WaveletChebConv
完整流程测试：数据集 -> 超图生成器 -> 小波切比雪夫卷积
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
from models.modules.wavelet_cheb_conv import WaveletChebConv


class WaveletPipelineTester:
    """Complete pipeline tester for Dataset -> HyperGenerator -> WaveletChebConv"""
    
    def __init__(self, hidden_dim=64, top_k=8, threshold=0.4, cheb_k=5, tau=0.5):
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.threshold = threshold
        self.cheb_k = cheb_k
        self.tau = tau
        
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
        
        # Wavelet Chebyshev convolution layers
        self.wavelet_conv1 = WaveletChebConv(
            in_dim=hidden_dim * 3,  # Multimodal features concatenated
            out_dim=hidden_dim,
            K=cheb_k,
            tau=tau
        )
        
        self.wavelet_conv2 = WaveletChebConv(
            in_dim=hidden_dim,
            out_dim=hidden_dim // 2,
            K=cheb_k,
            tau=tau
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 10)  # Assuming max 10 classes
        )
    
    def extract_multimodal_features(self, images, labels):
        """Extract multimodal features from images and labels"""
        batch_size = images.shape[0]
        
        # Process image dimensions
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            # Image features
            img_features = self.image_encoder(images)
            
            # Simulate text features (class-dependent patterns)
            text_features = torch.zeros(batch_size, self.hidden_dim)
            for i, label in enumerate(labels):
                base_pattern = torch.randn(self.hidden_dim) * 0.1
                class_pattern = torch.zeros(self.hidden_dim)
                if label.item() * 8 + 8 <= self.hidden_dim:
                    class_pattern[label.item() * 8:(label.item() + 1) * 8] = 1.0
                text_features[i] = base_pattern + class_pattern * 0.5
            
            text_features = self.text_encoder(text_features)
            
            # Simulate signal features (image statistics)
            signal_features = torch.zeros(batch_size, self.hidden_dim)
            for i in range(batch_size):
                img = images[i]
                stats = torch.tensor([
                    img.mean(), img.std(), img.max(), img.min(),
                    img.var(), img.median(), img.sum(), img.numel()
                ])
                # Repeat stats to fill the feature vector
                repeats = self.hidden_dim // len(stats)
                signal_pattern = stats.repeat(repeats)
                if len(signal_pattern) < self.hidden_dim:
                    padding = torch.randn(self.hidden_dim - len(signal_pattern)) * 0.1
                    signal_pattern = torch.cat([signal_pattern, padding])
                signal_features[i] = signal_pattern[:self.hidden_dim]
            
            signal_features = self.signal_encoder(signal_features)
        
        return img_features, text_features, signal_features
    
    def compute_hypergraph_laplacian(self, H, edge_weights=None):
        """Compute normalized hypergraph Laplacian matrix"""
        n_nodes, n_edges = H.shape
        
        # Node degrees
        if edge_weights is not None:
            # Weighted degree
            Dv = torch.diag(torch.sum(H * edge_weights.unsqueeze(0), dim=1))
        else:
            Dv = torch.diag(torch.sum(H, dim=1))
        
        # Edge degrees
        De = torch.diag(torch.sum(H, dim=0))
        
        # Avoid division by zero
        Dv_diag = torch.diag(Dv)
        De_diag = torch.diag(De)
        
        # Handle zero degrees
        Dv_diag = torch.where(Dv_diag > 0, Dv_diag, torch.ones_like(Dv_diag))
        De_diag = torch.where(De_diag > 0, De_diag, torch.ones_like(De_diag))
        
        # Normalized Laplacian: L = I - D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        Dv_inv_sqrt = torch.diag(torch.pow(Dv_diag, -0.5))
        De_inv = torch.diag(torch.pow(De_diag, -1.0))
        
        # Compute the transition matrix
        W = torch.mm(torch.mm(torch.mm(Dv_inv_sqrt, H), De_inv), H.t())
        W = torch.mm(W, Dv_inv_sqrt)
        
        # Laplacian matrix
        I = torch.eye(n_nodes, device=H.device)
        L = I - W
        
        return L, Dv, De
    
    def test_wavelet_pipeline(self, dataset_name='pathmnist', num_samples=64):
        """Test complete pipeline from dataset to wavelet convolution"""
        print(f"\n{'='*80}")
        print(f"Testing Wavelet Pipeline - Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
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
        
        try:
            # Step 1: Extract multimodal features
            print("\n🔄 Step 1: Extracting multimodal features...")
            img_features, text_features, signal_features = self.extract_multimodal_features(
                all_images, all_labels
            )
            
            print(f"Feature shapes:")
            print(f"  Image features: {img_features.shape}")
            print(f"  Text features: {text_features.shape}")
            print(f"  Signal features: {signal_features.shape}")
            
            # Step 2: Generate hypergraph
            print("\n🔄 Step 2: Generating hypergraph...")
            H, edge_weights = self.hypergraph_generator([img_features, text_features, signal_features])
            
            print(f"Hypergraph generated:")
            print(f"  Incidence matrix shape: {H.shape}")
            print(f"  Active hyperedges: {len(edge_weights)}")
            print(f"  Average edge weight: {edge_weights.mean().item():.4f}")
            
            # Step 3: Compute hypergraph Laplacian
            print("\n🔄 Step 3: Computing hypergraph Laplacian...")
            L, Dv, De = self.compute_hypergraph_laplacian(H, edge_weights)
            
            print(f"Laplacian matrix shape: {L.shape}")
            print(f"Laplacian eigenvalue range: [{L.min().item():.4f}, {L.max().item():.4f}]")
            
            # Step 4: Prepare node features for wavelet convolution
            print("\n🔄 Step 4: Preparing node features...")
            # Concatenate multimodal features as node features
            node_features = torch.cat([img_features, text_features, signal_features], dim=1)
            print(f"Node features shape: {node_features.shape}")
            
            # Step 5: Apply wavelet convolution layers
            print("\n🔄 Step 5: Applying wavelet convolution...")
            
            # First wavelet conv layer
            x1 = self.wavelet_conv1(node_features, L)
            x1 = F.relu(x1)
            print(f"After first wavelet conv: {x1.shape}")
            
            # Second wavelet conv layer
            x2 = self.wavelet_conv2(x1, L)
            x2 = F.relu(x2)
            print(f"After second wavelet conv: {x2.shape}")
            
            # Final classification
            logits = self.classifier(x2)
            predictions = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)
            
            print(f"Final output shape: {logits.shape}")
            print(f"Predicted classes: {predicted_classes[:10].tolist()}")
            print(f"Actual classes: {all_labels[:10].tolist()}")
            
            # Step 6: Compute accuracy
            accuracy = (predicted_classes == all_labels).float().mean()
            print(f"\n📊 Classification accuracy: {accuracy.item():.4f}")
            
            return {
                'dataset_name': dataset_name,
                'node_features': node_features.detach().numpy(),
                'hypergraph_H': H.detach().numpy(),
                'edge_weights': edge_weights.detach().numpy(),
                'laplacian': L.detach().numpy(),
                'conv1_output': x1.detach().numpy(),
                'conv2_output': x2.detach().numpy(),
                'logits': logits.detach().numpy(),
                'predictions': predictions.detach().numpy(),
                'predicted_classes': predicted_classes.detach().numpy(),
                'labels': all_labels.numpy(),
                'accuracy': accuracy.item(),
                'images': all_images.numpy()
            }
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_wavelet_results(self, results, save_dir='wavelet_test_results'):
        """Visualize wavelet pipeline results"""
        if results is None:
            print("❌ No results to visualize")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = results['dataset_name']
        
        print(f"\n📊 Creating wavelet pipeline visualizations...")
        
        # 1. Hypergraph structure analysis
        self._plot_hypergraph_analysis(results, save_dir)
        
        # 2. Laplacian eigenvalue analysis
        self._plot_laplacian_analysis(results, save_dir)
        
        # 3. Wavelet convolution feature evolution
        self._plot_feature_evolution(results, save_dir)
        
        # 4. Classification results
        self._plot_classification_results(results, save_dir)
        
        print(f"✅ Visualization completed, results saved in '{save_dir}' directory")
    
    def _plot_hypergraph_analysis(self, results, save_dir):
        """Plot hypergraph structure analysis"""
        H = results['hypergraph_H']
        edge_weights = results['edge_weights']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{results["dataset_name"].upper()} - Hypergraph Structure Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Incidence matrix heatmap
        im1 = axes[0, 0].imshow(H, cmap='Blues', aspect='auto')
        axes[0, 0].set_title('Hypergraph Incidence Matrix', fontweight='bold')
        axes[0, 0].set_xlabel('Hyperedges')
        axes[0, 0].set_ylabel('Nodes')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Node degree distribution
        node_degrees = H.sum(axis=1)
        axes[0, 1].hist(node_degrees, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Node Degree Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Node Degree')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hyperedge size distribution
        edge_sizes = H.sum(axis=0)
        axes[1, 0].hist(edge_sizes, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Hyperedge Size Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Hyperedge Size')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Edge weight distribution
        axes[1, 1].hist(edge_weights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Hyperedge Weight Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Weight')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/hypergraph_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_laplacian_analysis(self, results, save_dir):
        """Plot Laplacian matrix analysis"""
        L = results['laplacian']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{results["dataset_name"].upper()} - Laplacian Matrix Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Laplacian matrix heatmap
        im1 = axes[0].imshow(L, cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_title('Laplacian Matrix', fontweight='bold')
        axes[0].set_xlabel('Node Index')
        axes[0].set_ylabel('Node Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Eigenvalue distribution
        try:
            eigenvalues = np.linalg.eigvals(L)
            eigenvalues = np.real(eigenvalues)  # Take real part
            eigenvalues = np.sort(eigenvalues)
            
            axes[1].plot(eigenvalues, 'o-', markersize=3, linewidth=1)
            axes[1].set_title('Laplacian Eigenvalues', fontweight='bold')
            axes[1].set_xlabel('Index')
            axes[1].set_ylabel('Eigenvalue')
            axes[1].grid(True, alpha=0.3)
            
            # Eigenvalue histogram
            axes[2].hist(eigenvalues, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[2].set_title('Eigenvalue Distribution', fontweight='bold')
            axes[2].set_xlabel('Eigenvalue')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Eigenvalue computation failed:\n{str(e)}', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[2].text(0.5, 0.5, f'Eigenvalue computation failed:\n{str(e)}', 
                        ha='center', va='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/laplacian_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_evolution(self, results, save_dir):
        """Plot feature evolution through wavelet convolution layers"""
        node_features = results['node_features']
        conv1_output = results['conv1_output']
        conv2_output = results['conv2_output']
        labels = results['labels']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{results["dataset_name"].upper()} - Feature Evolution Through Wavelet Convolution', 
                     fontsize=16, fontweight='bold')
        
        # PCA visualization for each layer
        from sklearn.decomposition import PCA
        
        features_list = [
            ('Input Features', node_features),
            ('After WaveletConv1', conv1_output),
            ('After WaveletConv2', conv2_output)
        ]
        
        for idx, (title, features) in enumerate(features_list):
            # PCA projection
            try:
                if features.shape[1] > 2:
                    pca = PCA(n_components=2)
                    features_2d = pca.fit_transform(features)
                    variance_ratio = pca.explained_variance_ratio_
                else:
                    features_2d = features
                    variance_ratio = [1.0, 1.0]
                
                # Scatter plot by class
                ax_scatter = axes[0, idx]
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.arange(len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax_scatter.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                     c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30)
                
                ax_scatter.set_title(f'{title}\n(PC1: {variance_ratio[0]:.2%}, PC2: {variance_ratio[1]:.2%})', 
                                   fontweight='bold')
                ax_scatter.legend(fontsize=8)
                ax_scatter.grid(True, alpha=0.3)
                
                # Feature magnitude distribution
                ax_hist = axes[1, idx]
                feature_norms = np.linalg.norm(features, axis=1)
                ax_hist.hist(feature_norms, bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax_hist.set_title(f'{title}\nFeature Magnitude Distribution', fontweight='bold')
                ax_hist.set_xlabel('L2 Norm')
                ax_hist.set_ylabel('Frequency')
                ax_hist.grid(True, alpha=0.3)
                
            except Exception as e:
                axes[0, idx].text(0.5, 0.5, f'Visualization failed:\n{str(e)}', 
                                ha='center', va='center', transform=axes[0, idx].transAxes)
                axes[1, idx].text(0.5, 0.5, f'Visualization failed:\n{str(e)}', 
                                ha='center', va='center', transform=axes[1, idx].transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_evolution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_classification_results(self, results, save_dir):
        """Plot classification results"""
        predictions = results['predictions']
        predicted_classes = results['predicted_classes']
        labels = results['labels']
        accuracy = results['accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{results["dataset_name"].upper()} - Classification Results (Accuracy: {accuracy:.4f})', 
                     fontsize=16, fontweight='bold')
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        unique_labels = np.unique(labels)
        cm = confusion_matrix(labels, predicted_classes, labels=unique_labels)
        
        im1 = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_ylabel('True Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                              ha="center", va="center",
                              color="white" if cm[i, j] > thresh else "black")
        
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Prediction confidence distribution
        max_probs = np.max(predictions, axis=1)
        axes[0, 1].hist(max_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Max Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Class-wise accuracy
        class_accuracies = []
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                class_acc = (predicted_classes[mask] == labels[mask]).mean()
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        axes[1, 0].bar(unique_labels, class_accuracies, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Class-wise Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prediction scatter (first 2 PCA components)
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pred_2d = pca.fit_transform(predictions)
            
            scatter = axes[1, 1].scatter(pred_2d[:, 0], pred_2d[:, 1], c=labels, 
                                       cmap='tab10', alpha=0.7, s=30)
            axes[1, 1].set_title('Prediction Space (PCA)', fontweight='bold')
            axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
            axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
            plt.colorbar(scatter, ax=axes[1, 1])
            
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'PCA visualization failed:\n{str(e)}', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/classification_results.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    print("🚀 Wavelet Pipeline Testing: Dataset -> HyperGenerator -> WaveletChebConv")
    print("="*80)
    
    # Initialize tester
    tester = WaveletPipelineTester(
        hidden_dim=64, 
        top_k=8, 
        threshold=0.4, 
        cheb_k=5, 
        tau=0.5
    )
    
    # Test different datasets
    datasets_to_test = ['pathmnist', 'bloodmnist']
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Testing dataset: {dataset_name.upper()}")
        
        # Run pipeline
        results = tester.test_wavelet_pipeline(dataset_name, num_samples=64)
        
        if results is not None:
            # Create visualizations
            save_dir = f'wavelet_test_results/{dataset_name}'
            tester.visualize_wavelet_results(results, save_dir)
            
            # Print summary statistics
            print(f"\n📈 {dataset_name.upper()} Pipeline Summary:")
            print(f"  Node features shape: {results['node_features'].shape}")
            print(f"  Hypergraph shape: {results['hypergraph_H'].shape}")
            print(f"  Active hyperedges: {len(results['edge_weights'])}")
            print(f"  Classification accuracy: {results['accuracy']:.4f}")
            print(f"  Laplacian eigenvalue range: [{results['laplacian'].min():.4f}, {results['laplacian'].max():.4f}]")
        
        print(f"\n" + "-"*60)
    
    print(f"\n🎉 Wavelet pipeline testing completed!")
    print(f"📁 Results saved in 'wavelet_test_results' directory")


if __name__ == "__main__":
    main()
