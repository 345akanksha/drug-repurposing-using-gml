import torch
import numpy as np
from pathlib import Path
from torch_geometric.utils import from_networkx
from ml_engine.data_preprocess import build_graph
from ml_engine.gnn_models import GCNModel, GraphSageModel, GATModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Create models directory if it doesn't exist
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

def prepare_data():
    """Prepare graph data for training"""
    print("Building graph...")
    G, num_drugs, num_diseases = build_graph()
    nodes = list(G.nodes())
    
    # Convert to PyTorch Geometric data
    data = from_networkx(G)
    
    # Extract node features
    features = np.array([G.nodes[n]['features'] for n in nodes], dtype=np.float32)
    data.x = torch.from_numpy(features)
    
    print(f"Feature shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    
    # Prepare labels: positive samples (existing edges)
    edges = list(G.edges())
    edge_labels = []
    edge_indices = []
    
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Positive samples
    for u, v in edges:
        edge_indices.append([node_to_idx[u], node_to_idx[v]])
        edge_labels.append(1)
    
    # Generate negative samples (drug-disease pairs that don't exist)
    drug_nodes = [n for n in nodes if G.nodes[n].get('type') == 'drug']
    disease_nodes = [n for n in nodes if G.nodes[n].get('type') == 'disease']
    
    print(f"Generating negative samples from {len(drug_nodes)} drugs and {len(disease_nodes)} diseases...")
    
    negative_samples = []
    for _ in range(len(edges)):
        d = random.choice(drug_nodes)
        dis = random.choice(disease_nodes)
        if not G.has_edge(d, dis):
            negative_samples.append((d, dis))
            edge_indices.append([node_to_idx[d], node_to_idx[dis]])
            edge_labels.append(0)
    
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float32)
    
    print(f"Total samples: {len(edge_labels)} (Positive: {sum(edge_labels)}, Negative: {len(edge_labels) - sum(edge_labels)})")
    
    return data, edge_indices, edge_labels

def train_model(model, model_name, data, edge_indices, labels, epochs=200, lr=0.01):
    """Train a GNN model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Split data
    idx_train, idx_test = train_test_split(
        range(len(labels)), 
        test_size=0.2, 
        stratify=labels.numpy(),
        random_state=42
    )
    
    best_auc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Get node embeddings
        embeddings = model(data.x, data.edge_index)
        
        # Compute edge scores for training samples
        train_edges = edge_indices[:, idx_train]
        train_labels = labels[idx_train]
        
        scores = (embeddings[train_edges[0]] * embeddings[train_edges[1]]).sum(dim=1)
        loss = loss_fn(scores, train_labels)
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                embeddings = model(data.x, data.edge_index)
                
                # Test set evaluation
                test_edges = edge_indices[:, idx_test]
                test_labels = labels[idx_test]
                test_scores = (embeddings[test_edges[0]] * embeddings[test_edges[1]]).sum(dim=1)
                test_pred = torch.sigmoid(test_scores)
                
                auc = roc_auc_score(test_labels.numpy(), test_pred.numpy())
                
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + 1
                    # Save best model
                    torch.save(model.state_dict(), MODEL_DIR / f'{model_name}.pth')
                
                print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Test AUC: {auc:.4f} | Best: {best_auc:.4f} (Epoch {best_epoch})")
    
    print(f"\n{model_name} training completed. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {MODEL_DIR / f'{model_name}.pth'}")
    
    return model, best_auc

def main():
    """Main training pipeline"""
    print("="*60)
    print("GNN Training Pipeline for Drug Repurposing")
    print("="*60)
    
    # Prepare data
    data, edge_indices, labels = prepare_data()
    in_channels = data.x.shape[1]
    
    print(f"\nInput features: {in_channels}")
    print(f"Hidden channels: 64")
    print(f"Output embedding: 32")
    
    results = {}
    
    # Train GCN
    gcn = GCNModel(in_channels=in_channels, hidden_channels=64, out_channels=32)
    gcn, gcn_auc = train_model(gcn, 'gcn', data, edge_indices, labels)
    results['GCN'] = gcn_auc
    
    # Train GraphSAGE
    sage = GraphSageModel(in_channels=in_channels, hidden_channels=64, out_channels=32)
    sage, sage_auc = train_model(sage, 'sage', data, edge_indices, labels)
    results['GraphSAGE'] = sage_auc
    
    # Train GAT
    gat = GATModel(in_channels=in_channels, hidden_channels=16, out_channels=32, heads=4)
    gat, gat_auc = train_model(gat, 'gat', data, edge_indices, labels)
    results['GAT'] = gat_auc
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for model_name, auc in results.items():
        print(f"{model_name:15s}: {auc:.4f}")
    print("="*60)
    print(f"\nAll models saved to: {MODEL_DIR}/")

if __name__ == '__main__':
    main()