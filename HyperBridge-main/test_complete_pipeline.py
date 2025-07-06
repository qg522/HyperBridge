#!/usr/bin/env python3
"""
Complete End-to-End Pipeline Test
完整端到端流程测试：数据读取 -> 特征提取 -> 超图生成 -> 小波切比雪夫卷积 -> 剪枝正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_dataset import load_medmnist
from models.modules.hyper_generator import HybridHyperedgeGenerator
from models.modules.wavelet_cheb_conv import WaveletChebConv
from models.modules.pruning_regularizer import SpectralCutRegularizer


class CompletePipelineTester:
    """Complete end-to-end pipeline tester"""
    
    def __init__(self, hidden_dim=64, top_k=8, threshold=0.4, cheb_k=5, tau=0.5):
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.threshold = threshold
        self.cheb_k = cheb_k
        self.tau = tau
        
        print(f"🔧 Initializing Complete Pipeline Tester")
        print(f"   Hidden dim: {hidden_dim}, Top-k: {top_k}, Threshold: {threshold}")
        print(f"   Chebyshev K: {cheb_k}, Tau: {tau}")
        
        # Step 1: Feature extractors for multimodal data
        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(192, hidden_dim),  # 3*8*8=192 for RGB images
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.text_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.signal_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Step 2: Hypergraph generator
        self.hypergraph_generator = HybridHyperedgeGenerator(
            num_modalities=3,
            input_dims=[hidden_dim, hidden_dim, hidden_dim],
            hidden_dim=hidden_dim,
            top_k=top_k,
            threshold=threshold
        )
        
        # Step 3: Wavelet Chebyshev convolution layers
        self.wavelet_conv1 = WaveletChebConv(
            in_dim=hidden_dim * 3,  # Combined features from all modalities
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
        
        # Step 4: Pruning regularizer
        self.pruning_regularizer = SpectralCutRegularizer(
            use_rayleigh=True,
            reduction='mean'
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 4, 10)  # Assume max 10 classes
        )
        
        print("✅ Pipeline components initialized successfully")
    
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
    
    def compute_hypergraph_laplacian(self, H):
        """Compute normalized hypergraph Laplacian matrix"""
        # Node degree matrix
        node_degrees = H.sum(dim=1)  # [N]
        Dv = torch.diag(node_degrees + 1e-8)
        
        # Hyperedge degree matrix
        edge_degrees = H.sum(dim=0)  # [E]
        De = torch.diag(edge_degrees + 1e-8)
        
        # Compute normalized Laplacian: L = I - Dv^(-1/2) H De^(-1) H^T Dv^(-1/2)
        Dv_inv_sqrt = torch.diag(torch.pow(torch.diag(Dv), -0.5))
        De_inv = torch.diag(1.0 / torch.diag(De))
        
        # Adjacency-like matrix: A = Dv^(-1/2) H De^(-1) H^T Dv^(-1/2)
        A = torch.matmul(torch.matmul(torch.matmul(Dv_inv_sqrt, H), De_inv), 
                        torch.matmul(H.t(), Dv_inv_sqrt))
        
        # Laplacian: L = I - A
        I = torch.eye(A.shape[0])
        L = I - A
        
        return L, Dv, De
    
    def run_complete_pipeline(self, dataset_name='pathmnist', num_samples=64):
        """Run the complete end-to-end pipeline"""
        print(f"\n{'='*100}")
        print(f"🚀 RUNNING COMPLETE END-TO-END PIPELINE - Dataset: {dataset_name.upper()}")
        print(f"{'='*100}")
        
        results = {}
        
        # Step 1: Data Loading
        print("\n📁 Step 1: Loading Dataset...")
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
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Samples: {all_images.shape[0]}")
        print(f"   Image shape: {all_images.shape[1:]}")
        print(f"   Classes: {torch.unique(all_labels).tolist()}")
        
        results['raw_data'] = {
            'images': all_images.detach().cpu().numpy(),
            'labels': all_labels.detach().cpu().numpy(),
            'num_samples': len(all_labels),
            'num_classes': len(torch.unique(all_labels))
        }
        
        # Step 2: Feature Extraction
        print("\n🔍 Step 2: Extracting Multimodal Features...")
        img_features, text_features, signal_features = self.extract_multimodal_features(
            all_images, all_labels
        )
        
        print(f"✅ Feature extraction completed")
        print(f"   Image features: {img_features.shape}")
        print(f"   Text features: {text_features.shape}")
        print(f"   Signal features: {signal_features.shape}")
        
        results['features'] = {
            'img_features': img_features.detach().cpu().numpy(),
            'text_features': text_features.detach().cpu().numpy(),
            'signal_features': signal_features.detach().cpu().numpy()
        }
        
        # Step 3: Hypergraph Generation
        print("\n🕸️  Step 3: Generating Hypergraph...")
        try:
            H, edge_weights = self.hypergraph_generator([img_features, text_features, signal_features])
            
            # Filter out inactive hyperedges
            active_mask = H.sum(dim=0) > 0
            H_active = H[:, active_mask]
            active_edge_weights = edge_weights[:len(active_mask)][active_mask[:len(edge_weights)]]
            
            print(f"✅ Hypergraph generated successfully")
            print(f"   Incidence matrix: {H_active.shape}")
            print(f"   Active hyperedges: {len(active_edge_weights)}")
            print(f"   Average edge weight: {active_edge_weights.mean().item():.4f}")
            
            results['hypergraph'] = {
                'H': H_active.detach().cpu().numpy(),
                'edge_weights': active_edge_weights.detach().cpu().numpy(),
                'num_nodes': H_active.shape[0],
                'num_edges': H_active.shape[1]
            }
            
        except Exception as e:
            print(f"❌ Hypergraph generation failed: {str(e)}")
            return None
        
        # Step 4: Compute Hypergraph Laplacian
        print("\n📐 Step 4: Computing Hypergraph Laplacian...")
        try:
            L, Dv, De = self.compute_hypergraph_laplacian(H_active)
            
            print(f"✅ Laplacian computed successfully")
            print(f"   Laplacian matrix: {L.shape}")
            print(f"   Average node degree: {torch.diag(Dv).mean().item():.2f}")
            print(f"   Average edge degree: {torch.diag(De).mean().item():.2f}")
            
            results['laplacian'] = {
                'L': L.detach().cpu().numpy(),
                'Dv': Dv.detach().cpu().numpy(),
                'De': De.detach().cpu().numpy()
            }
            
        except Exception as e:
            print(f"❌ Laplacian computation failed: {str(e)}")
            return None
        
        # Step 5: Wavelet Chebyshev Convolution
        print("\n🌊 Step 5: Applying Wavelet Chebyshev Convolution...")
        try:
            # Combine all modality features as initial node features
            combined_features = torch.cat([img_features, text_features, signal_features], dim=1)
            
            # First wavelet convolution layer
            conv1_output = self.wavelet_conv1(combined_features, L)
            conv1_output = F.relu(conv1_output)
            
            # Second wavelet convolution layer
            conv2_output = self.wavelet_conv2(conv1_output, L)
            conv2_output = F.relu(conv2_output)
            
            print(f"✅ Wavelet convolution completed")
            print(f"   Layer 1 output: {conv1_output.shape}")
            print(f"   Layer 2 output: {conv2_output.shape}")
            
            results['wavelet_conv'] = {
                'conv1_output': conv1_output.detach().cpu().numpy(),
                'conv2_output': conv2_output.detach().cpu().numpy()
            }
            
        except Exception as e:
            print(f"❌ Wavelet convolution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Step 6: Pruning Regularization
        print("\n✂️  Step 6: Applying Pruning Regularization...")
        try:
            pruning_loss = self.pruning_regularizer(conv2_output, H_active, Dv, De)
            
            print(f"✅ Pruning regularization applied")
            print(f"   Spectral cut loss: {pruning_loss.item():.6f}")
            
            results['pruning'] = {
                'loss': pruning_loss.item()
            }
            
        except Exception as e:
            print(f"❌ Pruning regularization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Step 7: Final Classification
        print("\n🎯 Step 7: Final Classification...")
        try:
            logits = self.classifier(conv2_output)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Compute accuracy
            correct = (predictions == all_labels).float().sum()
            accuracy = correct / len(all_labels)
            
            print(f"✅ Classification completed")
            print(f"   Output logits: {logits.shape}")
            print(f"   Accuracy: {accuracy.item():.4f}")
            
            results['classification'] = {
                'logits': logits.detach().cpu().numpy(),
                'probabilities': probabilities.detach().cpu().numpy(),
                'predictions': predictions.detach().cpu().numpy(),
                'accuracy': accuracy.item()
            }
            
        except Exception as e:
            print(f"❌ Classification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Compile final results
        results['summary'] = {
            'dataset_name': dataset_name,
            'num_samples': num_samples,
            'num_classes': len(torch.unique(all_labels)),
            'hypergraph_nodes': H_active.shape[0],
            'hypergraph_edges': H_active.shape[1],
            'pruning_loss': pruning_loss.item(),
            'final_accuracy': accuracy.item(),
            'pipeline_success': True
        }
        
        print(f"\n{'='*100}")
        print("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
        print(f"📊 Final Summary:")
        print(f"   Dataset: {dataset_name.upper()}")
        print(f"   Samples processed: {num_samples}")
        print(f"   Hypergraph structure: {H_active.shape[0]} nodes, {H_active.shape[1]} edges")
        print(f"   Spectral cut loss: {pruning_loss.item():.6f}")
        print(f"   Final accuracy: {accuracy.item():.4f}")
        print(f"{'='*100}")
        
        return results
    
    def visualize_complete_results(self, results, save_dir='complete_pipeline_results'):
        """Create comprehensive visualization of the complete pipeline"""
        if results is None:
            print("❌ No results to visualize")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        dataset_name = results['summary']['dataset_name']
        
        print(f"\n📊 Creating comprehensive pipeline visualization...")
        
        # Create mega visualization
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle(f'{dataset_name.upper()} - Complete End-to-End Pipeline Analysis', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        # 1. Raw data overview
        ax1 = plt.subplot(4, 6, 1)
        labels = results['raw_data']['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        bars = ax1.bar(unique_labels, counts, color='skyblue', alpha=0.8)
        ax1.set_title('Dataset Distribution', fontweight='bold', fontsize=10)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sample image
        ax2 = plt.subplot(4, 6, 2)
        if len(results['raw_data']['images']) > 0:
            sample_img = results['raw_data']['images'][0]
            if len(sample_img.shape) == 3:  # RGB or channels
                if sample_img.shape[0] == 3:  # Channel first
                    sample_img = sample_img.transpose(1, 2, 0)
                ax2.imshow(sample_img)
            else:  # Grayscale
                ax2.imshow(sample_img, cmap='gray')
            ax2.set_title('Sample Image', fontweight='bold', fontsize=10)
            ax2.axis('off')
        
        # 3. Feature dimensions
        ax3 = plt.subplot(4, 6, 3)
        feature_dims = [
            results['features']['img_features'].shape[1],
            results['features']['text_features'].shape[1],
            results['features']['signal_features'].shape[1]
        ]
        feature_names = ['Image', 'Text', 'Signal']
        bars = ax3.bar(feature_names, feature_dims, color=['red', 'green', 'blue'], alpha=0.7)
        ax3.set_title('Feature Dims', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Dimension')
        
        # 4. Hypergraph structure
        ax4 = plt.subplot(4, 6, 4)
        hypergraph_stats = [
            results['hypergraph']['num_nodes'],
            results['hypergraph']['num_edges']
        ]
        ax4.bar(['Nodes', 'Edges'], hypergraph_stats, color=['purple', 'orange'], alpha=0.7)
        ax4.set_title('Hypergraph Structure', fontweight='bold', fontsize=10)
        ax4.set_ylabel('Count')
        
        # 5. Hypergraph incidence matrix
        ax5 = plt.subplot(4, 6, 5)
        H = results['hypergraph']['H']
        if H.shape[1] > 0:
            # Sample for display
            max_display = 30
            H_display = H[:min(max_display, H.shape[0]), :min(max_display, H.shape[1])]
            im = ax5.imshow(H_display, cmap='Blues', aspect='auto')
            ax5.set_title(f'Incidence Matrix\n({H_display.shape[0]}x{H_display.shape[1]})', 
                         fontweight='bold', fontsize=10)
            ax5.set_xlabel('Hyperedges')
            ax5.set_ylabel('Nodes')
        else:
            ax5.text(0.5, 0.5, 'No Hyperedges', ha='center', va='center',
                    transform=ax5.transAxes)
            ax5.set_title('Incidence Matrix', fontweight='bold', fontsize=10)
        
        # 6. Edge weights
        ax6 = plt.subplot(4, 6, 6)
        edge_weights = results['hypergraph']['edge_weights']
        if len(edge_weights) > 0:
            ax6.hist(edge_weights, bins=15, color='lightgreen', alpha=0.7)
            ax6.set_title('Edge Weights', fontweight='bold', fontsize=10)
            ax6.set_xlabel('Weight')
            ax6.set_ylabel('Frequency')
        else:
            ax6.text(0.5, 0.5, 'No Weights', ha='center', va='center',
                    transform=ax6.transAxes)
            ax6.set_title('Edge Weights', fontweight='bold', fontsize=10)
        
        # 7. Laplacian eigenvalues (sample)
        ax7 = plt.subplot(4, 6, 7)
        try:
            L = results['laplacian']['L']
            eigenvals, _ = np.linalg.eigh(L)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
            ax7.plot(eigenvals[:20], 'o-', linewidth=2, markersize=4)
            ax7.set_title('Laplacian Eigenvalues', fontweight='bold', fontsize=10)
            ax7.set_xlabel('Index')
            ax7.set_ylabel('Eigenvalue')
            ax7.grid(True, alpha=0.3)
        except:
            ax7.text(0.5, 0.5, 'Eigenvalue\nComputation\nFailed', ha='center', va='center',
                    transform=ax7.transAxes)
            ax7.set_title('Laplacian Eigenvalues', fontweight='bold', fontsize=10)
        
        # 8. Wavelet convolution outputs
        ax8 = plt.subplot(4, 6, (8, 9))
        conv1_out = results['wavelet_conv']['conv1_output']
        conv2_out = results['wavelet_conv']['conv2_output']
        
        # Show feature activation heatmaps
        conv1_sample = conv1_out[:20, :20]  # Sample for visualization
        conv2_sample = conv2_out[:20, :20]
        
        # Plot both side by side
        im1 = ax8.imshow(np.concatenate([conv1_sample, conv2_sample], axis=1), 
                        cmap='viridis', aspect='auto')
        ax8.set_title('Wavelet Conv Outputs (Layer1|Layer2)', fontweight='bold', fontsize=10)
        ax8.set_xlabel('Feature Dimensions')
        ax8.set_ylabel('Samples')
        plt.colorbar(im1, ax=ax8, shrink=0.8)
        
        # 9. Feature evolution through layers
        ax9 = plt.subplot(4, 6, 10)
        layer_dims = [
            results['features']['img_features'].shape[1] * 3,  # Combined input
            results['wavelet_conv']['conv1_output'].shape[1],
            results['wavelet_conv']['conv2_output'].shape[1]
        ]
        layer_names = ['Input', 'Conv1', 'Conv2']
        ax9.plot(layer_names, layer_dims, 'o-', linewidth=3, markersize=8)
        ax9.set_title('Feature Dimension Evolution', fontweight='bold', fontsize=10)
        ax9.set_ylabel('Dimension')
        ax9.grid(True, alpha=0.3)
        for i, dim in enumerate(layer_dims):
            ax9.text(i, dim + 2, str(dim), ha='center', va='bottom', fontweight='bold')
        
        # 10. Classification results
        ax10 = plt.subplot(4, 6, 11)
        predictions = results['classification']['predictions']
        true_labels = results['raw_data']['labels']
        
        # Confusion matrix (simplified)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predictions)
        im = ax10.imshow(cm, cmap='Blues')
        ax10.set_title(f'Confusion Matrix\n(Acc: {results["classification"]["accuracy"]:.3f})', 
                      fontweight='bold', fontsize=10)
        ax10.set_xlabel('Predicted')
        ax10.set_ylabel('True')
        
        # 11. Pipeline metrics summary
        ax11 = plt.subplot(4, 6, 12)
        ax11.axis('off')
        summary_text = f"""
PIPELINE SUMMARY
Dataset: {dataset_name.upper()}
Samples: {results['summary']['num_samples']}
Classes: {results['summary']['num_classes']}

Hypergraph:
  Nodes: {results['summary']['hypergraph_nodes']}
  Edges: {results['summary']['hypergraph_edges']}
  
Spectral Loss: {results['summary']['pruning_loss']:.4f}
Final Accuracy: {results['summary']['final_accuracy']:.4f}

STATUS: SUCCESS ✅
        """
        ax11.text(0.05, 0.95, summary_text.strip(), transform=ax11.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # 12-18. Feature space visualizations (t-SNE)
        feature_types = [
            ('img_features', 'Image Features'),
            ('text_features', 'Text Features'), 
            ('signal_features', 'Signal Features')
        ]
        
        for idx, (feat_key, feat_title) in enumerate(feature_types):
            ax = plt.subplot(4, 6, 13 + idx)
            try:
                features = results['features'][feat_key]
                if features.shape[1] > 2:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]//4))
                    features_2d = tsne.fit_transform(features)
                else:
                    features_2d = features
                
                scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                                   c=labels, cmap='tab10', alpha=0.7, s=20)
                ax.set_title(f'{feat_title}\n(t-SNE)', fontweight='bold', fontsize=10)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'Visualization\nFailed', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{feat_title}', fontweight='bold', fontsize=10)
        
        # 19. Wavelet conv layer 1 t-SNE
        ax19 = plt.subplot(4, 6, 16)
        try:
            conv1_features = results['wavelet_conv']['conv1_output']
            if conv1_features.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, conv1_features.shape[0]//4))
                conv1_2d = tsne.fit_transform(conv1_features)
            else:
                conv1_2d = conv1_features
            
            scatter = ax19.scatter(conv1_2d[:, 0], conv1_2d[:, 1], 
                                 c=labels, cmap='tab10', alpha=0.7, s=20)
            ax19.set_title('Wavelet Conv1\n(t-SNE)', fontweight='bold', fontsize=10)
            ax19.grid(True, alpha=0.3)
        except:
            ax19.text(0.5, 0.5, 'Conv1\nVisualization\nFailed', ha='center', va='center',
                     transform=ax19.transAxes)
            ax19.set_title('Wavelet Conv1', fontweight='bold', fontsize=10)
        
        # 20. Wavelet conv layer 2 t-SNE
        ax20 = plt.subplot(4, 6, 17)
        try:
            conv2_features = results['wavelet_conv']['conv2_output']
            if conv2_features.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, conv2_features.shape[0]//4))
                conv2_2d = tsne.fit_transform(conv2_features)
            else:
                conv2_2d = conv2_features
            
            scatter = ax20.scatter(conv2_2d[:, 0], conv2_2d[:, 1], 
                                 c=labels, cmap='tab10', alpha=0.7, s=20)
            ax20.set_title('Wavelet Conv2\n(t-SNE)', fontweight='bold', fontsize=10)
            ax20.grid(True, alpha=0.3)
        except:
            ax20.text(0.5, 0.5, 'Conv2\nVisualization\nFailed', ha='center', va='center',
                     transform=ax20.transAxes)
            ax20.set_title('Wavelet Conv2', fontweight='bold', fontsize=10)
        
        # 21. Node degree distribution
        ax21 = plt.subplot(4, 6, 18)
        if H.shape[1] > 0:
            node_degrees = H.sum(axis=1)
            ax21.hist(node_degrees, bins=15, color='lightcoral', alpha=0.7)
            ax21.set_title('Node Degrees', fontweight='bold', fontsize=10)
            ax21.set_xlabel('Degree')
            ax21.set_ylabel('Frequency')
            ax21.grid(True, alpha=0.3)
        else:
            ax21.text(0.5, 0.5, 'No Degree\nData', ha='center', va='center',
                     transform=ax21.transAxes)
            ax21.set_title('Node Degrees', fontweight='bold', fontsize=10)
        
        # 22. Edge degree distribution  
        ax22 = plt.subplot(4, 6, 19)
        if H.shape[1] > 0:
            edge_degrees = H.sum(axis=0)
            ax22.hist(edge_degrees, bins=15, color='gold', alpha=0.7)
            ax22.set_title('Edge Degrees', fontweight='bold', fontsize=10)
            ax22.set_xlabel('Degree')
            ax22.set_ylabel('Frequency')
            ax22.grid(True, alpha=0.3)
        else:
            ax22.text(0.5, 0.5, 'No Edge\nData', ha='center', va='center',
                     transform=ax22.transAxes)
            ax22.set_title('Edge Degrees', fontweight='bold', fontsize=10)
        
        # 23. Classification probabilities
        ax23 = plt.subplot(4, 6, 20)
        probabilities = results['classification']['probabilities']
        max_probs = np.max(probabilities, axis=1)
        ax23.hist(max_probs, bins=15, color='pink', alpha=0.7)
        ax23.set_title('Max Class Probabilities', fontweight='bold', fontsize=10)
        ax23.set_xlabel('Probability')
        ax23.set_ylabel('Frequency')
        ax23.grid(True, alpha=0.3)
        
        # 24. Final pipeline flow diagram
        ax24 = plt.subplot(4, 6, (21, 24))
        ax24.axis('off')
        
        # Draw pipeline flow
        pipeline_steps = [
            'Data\nLoading', 'Feature\nExtraction', 'Hypergraph\nGeneration',
            'Laplacian\nComputation', 'Wavelet\nConvolution', 'Pruning\nRegularization',
            'Classification'
        ]
        
        # Create a simple flow diagram
        y_pos = 0.8
        step_width = 1.0 / len(pipeline_steps)
        
        for i, step in enumerate(pipeline_steps):
            x_pos = (i + 0.5) * step_width
            
            # Draw box
            bbox = dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7)
            ax24.text(x_pos, y_pos, step, ha='center', va='center',
                     transform=ax24.transAxes, fontsize=8, fontweight='bold',
                     bbox=bbox)
            
            # Draw arrow to next step
            if i < len(pipeline_steps) - 1:
                ax24.annotate('', xy=((i + 1.3) * step_width, y_pos), 
                            xytext=((i + 0.7) * step_width, y_pos),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Add success indicator
        ax24.text(0.5, 0.3, '🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉', 
                 ha='center', va='center', transform=ax24.transAxes,
                 fontsize=14, fontweight='bold', color='green')
        
        ax24.set_title('Complete Pipeline Flow', fontweight='bold', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_dir}/complete_pipeline_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive visualization saved to '{save_dir}'")


def main():
    """Main function to run the complete end-to-end pipeline test"""
    print("🚀 COMPLETE END-TO-END PIPELINE TEST")
    print("="*100)
    print("Pipeline: Data Loading → Feature Extraction → Hypergraph Generation")
    print("         → Laplacian Computation → Wavelet Convolution → Pruning → Classification")
    print("="*100)
    
    # Initialize complete pipeline tester
    tester = CompletePipelineTester(
        hidden_dim=64,
        top_k=8, 
        threshold=0.4,
        cheb_k=5,
        tau=0.5
    )
    
    # Test datasets
    datasets_to_test = ['pathmnist', 'bloodmnist']
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Testing complete pipeline with {dataset_name.upper()}")
        
        # Run complete pipeline
        results = tester.run_complete_pipeline(dataset_name, num_samples=64)
        
        if results is not None:
            # Create comprehensive visualization
            save_dir = f'complete_pipeline_results/{dataset_name}'
            tester.visualize_complete_results(results, save_dir)
            
            print(f"\n🏆 {dataset_name.upper()} COMPLETE PIPELINE SUCCESS!")
            print(f"  📊 Dataset: {results['summary']['num_samples']} samples, {results['summary']['num_classes']} classes")
            print(f"  🕸️  Hypergraph: {results['summary']['hypergraph_nodes']} nodes, {results['summary']['hypergraph_edges']} edges")
            print(f"  ✂️  Spectral loss: {results['summary']['pruning_loss']:.6f}")
            print(f"  🎯 Final accuracy: {results['summary']['final_accuracy']:.4f}")
        else:
            print(f"  ❌ Complete pipeline failed for {dataset_name}")
        
        print(f"\n" + "="*80)
    
    print(f"\n🎉 COMPLETE END-TO-END PIPELINE TESTING FINISHED!")
    print(f"📁 All results saved in 'complete_pipeline_results' directory")
    print(f"🔍 Check the comprehensive visualizations for detailed analysis")


if __name__ == "__main__":
    main()
