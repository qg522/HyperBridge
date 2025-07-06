#!/usr/bin/env python3
"""
Complete HyperBridge Training Pipeline
完整的HyperBridge训练流程：数据加载 -> 超图生成 -> 小波卷积 -> 剪枝正则化 -> 训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_dataset import load_medmnist
from models.modules.hyper_generator import HybridHyperedgeGenerator
from models.modules.wavelet_cheb_conv import WaveletChebConv
from models.modules.pruning_regularizer import SpectralCutRegularizer


class HyperBridgeModel(nn.Module):
    """Complete HyperBridge Model with all components"""
    
    def __init__(self, config):
        super(HyperBridgeModel, self).__init__()
        self.config = config
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Feature extractors
        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(192, config['hidden_dim']),  # 3*8*8=192
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['hidden_dim'], config['hidden_dim'])
        ).to(self.device)
        
        self.text_encoder = nn.Sequential(
            nn.Embedding(config['vocab_size'], config['embed_dim']),
            nn.LSTM(config['embed_dim'], config['text_hidden'], batch_first=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config['text_hidden'], config['hidden_dim'])
        ).to(self.device)
        
        self.signal_encoder = nn.Sequential(
            nn.Linear(config['signal_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['hidden_dim'], config['hidden_dim'])
        ).to(self.device)
        
        # Hypergraph generator
        self.hypergraph_generator = HybridHyperedgeGenerator(
            num_modalities=3,
            input_dims=[config['hidden_dim']] * 3,
            hidden_dim=config['hidden_dim'],
            top_k=config['top_k'],
            threshold=config['threshold']
        ).to(self.device)
        
        # Wavelet convolution layers
        self.wavelet_conv1 = WaveletChebConv(
            in_dim=config['hidden_dim'] * 3,  # Concatenated features
            out_dim=config['hidden_dim'],
            K=config['cheb_k'],
            tau=config['tau']
        ).to(self.device)
        
        self.wavelet_conv2 = WaveletChebConv(
            in_dim=config['hidden_dim'],
            out_dim=config['hidden_dim'] // 2,
            K=config['cheb_k'],
            tau=config['tau']
        ).to(self.device)
        
        # Pruning regularizer
        self.pruning_regularizer = SpectralCutRegularizer(
            use_rayleigh=True,
            reduction='mean'
        ).to(self.device)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(config['hidden_dim'] // 2, config['hidden_dim'] // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config['hidden_dim'] // 4, config['n_classes'])
        ).to(self.device)
        
    def forward(self, images, text, signals, return_hypergraph=False):
        batch_size = images.shape[0]
        
        # Ensure all inputs are on the correct device
        images = images.to(self.device)
        text = text.to(self.device)
        signals = signals.to(self.device)
        
        # Extract features from each modality
        img_features = self.image_encoder(images)
        
        # Text encoding with LSTM
        text_embedding = self.text_encoder[0](text)  # Embedding
        lstm_out, _ = self.text_encoder[1](text_embedding)  # LSTM
        text_features = lstm_out.mean(dim=1)  # Average pooling
        text_features = self.text_encoder[4](text_features)  # Final linear
        
        signal_features = self.signal_encoder(signals)
        
        # Ensure all feature tensors are on the same device
        img_features = img_features.to(self.device)
        text_features = text_features.to(self.device)
        signal_features = signal_features.to(self.device)
        
        # Generate hypergraph
        H, edge_weights = self.hypergraph_generator([img_features, text_features, signal_features])
        
        # Ensure all tensors are on the same device
        H = H.to(self.device)
        edge_weights = edge_weights.to(self.device)
        
        # Compute hypergraph Laplacian
        L = self._compute_laplacian(H, edge_weights)
        L = L.to(self.device)
        
        # Concatenate features as node features
        node_features = torch.cat([img_features, text_features, signal_features], dim=1)
        node_features = node_features.to(self.device)
        
        # Apply wavelet convolution layers
        x1 = self.wavelet_conv1(node_features, L)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = self.wavelet_conv2(x1, L)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        # Final classification
        logits = self.classifier(x2)
        
        # Compute pruning regularization loss
        Dv, De = self._compute_degree_matrices(H)
        Dv = Dv.to(self.device)
        De = De.to(self.device)
        
        pruning_loss = self.pruning_regularizer(x2, H, Dv, De)
        
        if return_hypergraph:
            return logits, pruning_loss, H, edge_weights, x2
        else:
            return logits, pruning_loss
    
    def _compute_laplacian(self, H, edge_weights=None):
        """Compute normalized hypergraph Laplacian"""
        n_nodes, n_edges = H.shape
        
        # Ensure H is on the correct device
        H = H.to(self.device)
        if edge_weights is not None:
            edge_weights = edge_weights.to(self.device)
        
        # Node and edge degrees
        if edge_weights is not None:
            Dv = torch.diag(torch.sum(H * edge_weights.unsqueeze(0), dim=1)).to(self.device)
        else:
            Dv = torch.diag(torch.sum(H, dim=1)).to(self.device)
        
        De = torch.diag(torch.sum(H, dim=0)).to(self.device)
        
        # Avoid division by zero
        Dv_diag = torch.diag(Dv) + 1e-8
        De_diag = torch.diag(De) + 1e-8
        
        # Normalized Laplacian
        Dv_inv_sqrt = torch.diag(torch.pow(Dv_diag, -0.5)).to(self.device)
        De_inv = torch.diag(torch.pow(De_diag, -1.0)).to(self.device)
        
        W = torch.mm(torch.mm(torch.mm(Dv_inv_sqrt, H), De_inv), H.t())
        W = torch.mm(W, Dv_inv_sqrt).to(self.device)
        
        I = torch.eye(n_nodes, device=self.device)
        L = I - W
        
        return L.to(self.device)
    
    def _compute_degree_matrices(self, H):
        """Compute degree matrices for pruning regularizer"""
        node_degrees = H.sum(dim=1) + 1e-8
        edge_degrees = H.sum(dim=0) + 1e-8
        
        Dv = torch.diag(node_degrees).to(self.device)
        De = torch.diag(edge_degrees).to(self.device)
        
        return Dv, De


class HyperBridgeTrainer:
    """Complete training pipeline for HyperBridge"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Store device in config for model access
        self.config['device'] = self.device
        
        # Initialize model
        self.model = HyperBridgeModel(config).to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config['lr_step'], 
            gamma=config['lr_gamma']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def prepare_data(self, dataset_name='pathmnist', num_samples=None):
        """Prepare training and validation data"""
        print("Loading and preparing data...")
        
        # Load MedMNIST data
        data_loaders = load_medmnist(batch_size=self.config['batch_size'])
        
        if dataset_name not in data_loaders:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        train_loader = data_loaders[dataset_name]['train']
        val_loader = data_loaders[dataset_name]['val']
        
        # Convert to our format
        train_data = self._extract_data(train_loader, num_samples)
        val_data = self._extract_data(val_loader, num_samples//2 if num_samples else None)
        
        # Create dataloaders
        train_dataset = TensorDataset(*train_data)
        val_dataset = TensorDataset(*val_data)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_data, val_data
    
    def _extract_data(self, dataloader, num_samples=None):
        """Extract and preprocess data from dataloader"""
        all_images = []
        all_labels = []
        collected = 0
        
        for images, labels in dataloader:
            all_images.append(images)
            all_labels.append(labels.squeeze())
            collected += images.shape[0]
            
            if num_samples and collected >= num_samples:
                break
        
        images = torch.cat(all_images)
        labels = torch.cat(all_labels)
        
        if num_samples:
            images = images[:num_samples]
            labels = labels[:num_samples]
        
        batch_size = images.shape[0]
        
        # Process images
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Create simulated text data
        text_data = torch.randint(
            1, self.config['vocab_size'], 
            (batch_size, self.config['text_seq_len'])
        )
        
        # Create simulated signal data
        signal_data = torch.randn(batch_size, self.config['signal_dim'])
        
        return images, text_data, signal_data, labels
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_pruning_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, text, signals, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            text = text.to(self.device)
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, pruning_loss = self.model(images, text, signals)
            
            # Compute losses
            task_loss = self.criterion(logits, labels)
            total_loss_batch = task_loss + self.config['lambda_struct'] * pruning_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            total_pruning_loss += pruning_loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {total_loss_batch.item():.4f}, '
                      f'Task Loss: {task_loss.item():.4f}, '
                      f'Pruning Loss: {pruning_loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        avg_pruning_loss = total_pruning_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        print(f'Epoch {epoch} Training - Loss: {avg_loss:.4f}, '
              f'Pruning Loss: {avg_pruning_loss:.6f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, text, signals, labels in self.val_loader:
                images = images.to(self.device)
                text = text.to(self.device)
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                logits, pruning_loss = self.model(images, text, signals)
                
                task_loss = self.criterion(logits, labels)
                total_loss_batch = task_loss + self.config['lambda_struct'] * pruning_loss
                
                total_loss += total_loss_batch.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch} Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, num_epochs, save_dir='training_results'):
        """Main training loop"""
        print(f"\n{'='*80}")
        print("🚀 STARTING HYPERBRIDGE TRAINING")
        print(f"{'='*80}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_accuracy = 0
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(self.model.state_dict(), f"{save_dir}/best_model.pth")
                print(f"✅ New best validation accuracy: {best_val_accuracy:.2f}%")
        
        print(f"\n{'='*80}")
        print("🎉 TRAINING COMPLETED!")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        print(f"{'='*80}")
        
        return best_val_accuracy
    
    def evaluate_and_visualize(self, save_dir='training_results'):
        """Comprehensive evaluation and visualization"""
        print("\n📊 Creating comprehensive evaluation and visualizations...")
        
        # Load best model
        self.model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
        self.model.eval()
        
        # Detailed validation
        all_predictions = []
        all_labels = []
        all_embeddings = []
        hypergraph_info = []
        
        with torch.no_grad():
            for images, text, signals, labels in self.val_loader:
                images = images.to(self.device)
                text = text.to(self.device)
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                logits, pruning_loss, H, edge_weights, embeddings = self.model(
                    images, text, signals, return_hypergraph=True
                )
                
                _, predicted = torch.max(logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_embeddings.append(embeddings.cpu())
                
                # Store hypergraph info
                hypergraph_info.append({
                    'H': H.cpu(),
                    'edge_weights': edge_weights.cpu(),
                    'node_count': H.shape[0],
                    'edge_count': H.shape[1]
                })
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Calculate detailed metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Print metrics
        print("\n📈 FINAL EVALUATION METRICS:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Create visualizations
        self._create_comprehensive_visualizations(
            all_labels, all_predictions, all_embeddings, 
            hypergraph_info, save_dir
        )
        
        return metrics
    
    def _create_comprehensive_visualizations(self, labels, predictions, embeddings, 
                                           hypergraph_info, save_dir):
        """Create comprehensive visualizations"""
        
        # 1. Training curves
        plt.figure(figsize=(15, 10))
        
        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(2, 3, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', linewidth=2)
        plt.plot(self.val_accuracies, label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion matrix
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Embeddings visualization
        plt.subplot(2, 3, 4)
        if embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
            embeddings_2d = tsne.fit_transform(embeddings.numpy())
        else:
            embeddings_2d = embeddings.numpy()
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.arange(len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', alpha=0.7, s=30)
        
        plt.title('Learned Embeddings (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # Hypergraph statistics
        plt.subplot(2, 3, 5)
        node_counts = [info['node_count'] for info in hypergraph_info]
        edge_counts = [info['edge_count'] for info in hypergraph_info]
        
        plt.hist(edge_counts, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Hyperedge Count Distribution')
        plt.xlabel('Number of Hyperedges')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Edge weight distribution
        plt.subplot(2, 3, 6)
        all_edge_weights = []
        for info in hypergraph_info:
            all_edge_weights.extend(info['edge_weights'].numpy())
        
        if all_edge_weights:
            plt.hist(all_edge_weights, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.title('Hyperedge Weight Distribution')
            plt.xlabel('Edge Weight')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/comprehensive_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved to {save_dir}/comprehensive_results.png")


def main():
    """Main function"""
    # Configuration
    config = {
        # Model architecture
        'hidden_dim': 64,
        'n_classes': 9,  # Adjust based on dataset
        'vocab_size': 1000,
        'embed_dim': 128,
        'text_hidden': 32,
        'text_seq_len': 20,
        'signal_dim': 64,
        
        # Hypergraph parameters
        'top_k': 8,
        'threshold': 0.4,
        
        # Wavelet parameters
        'cheb_k': 5,
        'tau': 0.5,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'lambda_struct': 0.1,  # Pruning regularization weight
        'lr_step': 20,
        'lr_gamma': 0.5,
    }
    
    print("🚀 HyperBridge Complete Pipeline Training")
    print("="*60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Initialize trainer
    trainer = HyperBridgeTrainer(config)
    
    # Test different datasets
    datasets_to_test = ['pathmnist', 'bloodmnist']
    
    for dataset_name in datasets_to_test:
        print(f"\n🔬 Training on dataset: {dataset_name.upper()}")
        
        try:
            # Prepare data
            train_data, val_data = trainer.prepare_data(dataset_name, num_samples=256)
            
            # Adjust number of classes based on data
            unique_classes = len(torch.unique(train_data[3]))
            config['n_classes'] = unique_classes
            print(f"Detected {unique_classes} classes in {dataset_name}")
            
            # Reinitialize model with correct number of classes
            trainer = HyperBridgeTrainer(config)
            trainer.prepare_data(dataset_name, num_samples=256)
            
            # Train
            save_dir = f'hyperbridge_results/{dataset_name}'
            best_accuracy = trainer.train(num_epochs=30, save_dir=save_dir)
            
            # Evaluate and visualize
            final_metrics = trainer.evaluate_and_visualize(save_dir)
            
            print(f"\n📊 {dataset_name.upper()} FINAL RESULTS:")
            print(f"  Best validation accuracy: {best_accuracy:.2f}%")
            print(f"  Final test metrics: {final_metrics}")
            
        except Exception as e:
            print(f"❌ Error training on {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "-"*80)
    
    print(f"\n🎉 Complete pipeline training finished!")
    print(f"📁 Results saved in 'hyperbridge_results' directory")


if __name__ == "__main__":
    main()
