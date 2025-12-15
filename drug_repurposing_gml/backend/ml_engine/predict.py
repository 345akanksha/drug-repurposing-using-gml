# ml_engine/predict.py
import json
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import SpectralEmbedding
from ml_engine.data_preprocess import build_graph
import ml_engine.gnn_models as gnn_models
import torch
from torch_geometric.utils import from_networkx
from networkx.readwrite import json_graph

MODEL_DIR = Path('models')

def compute_embeddings_fallback(G, dim=32):
    try:
        A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
        n_components = min(dim, max(2, A.shape[0] - 1))
        se = SpectralEmbedding(n_components=n_components, affinity='precomputed')
        emb = se.fit_transform(A.toarray())
        if emb.shape[1] < dim:
            emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])), 'constant')
        return emb.astype(np.float32)
    except Exception as e:
        print(f"Spectral embedding failed: {e}")
        return np.random.randn(G.number_of_nodes(), dim).astype(np.float32)

def to_serializable(obj):
    """
    Helper for json.dumps default param to turn numpy scalars/arrays into Python natives.
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # fallback
    raise TypeError(f"Type {type(obj)} not serializable")

def run_predictions(model_type='ensemble', top_k=20, max_nodes_viz=200):
    """
    Runs models (GCN, GraphSAGE, GAT) to compute embeddings and predicted drug-disease pairs.

    Returns a dict with keys:
      - 'graph_obj': original networkx Graph (for internal use; NOT JSON-serializable)
      - 'predictions_list': full list of all candidate predictions (internal)
      - 'response': JSON-serializable dict intended for API responses (safe to send)
    """
    print(f"\n{'='*60}")
    print(f"Running predictions with model: {model_type.upper()}")
    print(f"{'='*60}")

    # Build graph and node list
    G, num_drugs, num_diseases = build_graph()
    nodes = list(G.nodes())
    print(f"Graph: {num_drugs} drugs, {num_diseases} diseases, {G.number_of_edges()} edges")

    # Compute embeddings using GNNs (fall back to spectral if needed)
    try:
        data = from_networkx(G)
        # assemble features in node order
        features = np.array([G.nodes[n]['features'] for n in nodes], dtype=np.float32)
        x = torch.from_numpy(features)
        data.x = x
        edge_index = data.edge_index
        in_channels = data.x.shape[1]
        emb_list = []

        if model_type in ['gcn', 'ensemble']:
            print("Loading GCN model...")
            gcn = gnn_models.GCNModel(in_channels=in_channels, hidden_channels=64, out_channels=32)
            model_path = MODEL_DIR / 'gcn.pth'
            if model_path.exists():
                gcn.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Loaded trained GCN from {model_path}")
            else:
                print(f"Warning: No trained model at {model_path}, using untrained GCN")
            gcn.eval()
            with torch.no_grad():
                emb_gcn = gcn(data.x, edge_index).numpy()
            emb_list.append(emb_gcn)
            print(f"GCN embeddings computed: {emb_gcn.shape}")

        if model_type in ['sage', 'ensemble']:
            print("Loading GraphSAGE model...")
            sage = gnn_models.GraphSageModel(in_channels=in_channels, hidden_channels=64, out_channels=32)
            model_path = MODEL_DIR / 'sage.pth'
            if model_path.exists():
                sage.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Loaded trained GraphSAGE from {model_path}")
            else:
                print(f"Warning: No trained model at {model_path}, using untrained GraphSAGE")
            sage.eval()
            with torch.no_grad():
                emb_sage = sage(data.x, edge_index).numpy()
            emb_list.append(emb_sage)
            print(f"GraphSAGE embeddings computed: {emb_sage.shape}")

        if model_type in ['gat', 'ensemble']:
            print("Loading GAT model...")
            gat = gnn_models.GATModel(in_channels=in_channels, hidden_channels=16, out_channels=32, heads=4)
            model_path = MODEL_DIR / 'gat.pth'
            if model_path.exists():
                gat.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Loaded trained GAT from {model_path}")
            else:
                print(f"Warning: No trained model at {model_path}, using untrained GAT")
            gat.eval()
            with torch.no_grad():
                emb_gat = gat(data.x, edge_index).numpy()
            emb_list.append(emb_gat)
            print(f"GAT embeddings computed: {emb_gat.shape}")

        # average embeddings across models present
        emb = np.mean(emb_list, axis=0)
        print(f"Final embeddings shape: {emb.shape}")

    except Exception as e:
        print(f"GNN failed ({e}), using fallback spectral embeddings...")
        emb = compute_embeddings_fallback(G, dim=32)

    # Split nodes into drug/disease sets and index mapping
    drug_nodes = [n for n in nodes if G.nodes[n].get('type') == 'drug']
    disease_nodes = [n for n in nodes if G.nodes[n].get('type') == 'disease']
    idx = {n: i for i, n in enumerate(nodes)}

    print(f"Computing similarity for {len(drug_nodes)} x {len(disease_nodes)} pairs...")
    sim = cosine_similarity(emb)  # full square similarity matrix

    # Build candidate predictions for drug-disease pairs that are NOT already connected
    preds = []
    for d in drug_nodes:
        for dis in disease_nodes:
            i, j = idx[d], idx[dis]
            if not G.has_edge(d, dis):
                drug_label = G.nodes[d].get('label', d)
                disease_label = G.nodes[dis].get('label', dis)
                preds.append({
                    'drug': d,
                    'drug_name': drug_label,
                    'disease': dis,
                    'disease_name': disease_label,
                    'score': float(sim[i, j])
                })

    # Sort by score desc and pick top_k unique drugs (one top disease per drug)
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    seen_drugs = set()
    unique_preds = []
    for pred in preds:
        if pred['drug'] not in seen_drugs:
            unique_preds.append(pred)
            seen_drugs.add(pred['drug'])
        if len(unique_preds) == top_k:
            break

    # For the chosen drugs get all known associations (out-edges from the drug)
    known_edges = []
    known_nodes = set()
    for pred in unique_preds:
        drug = pred['drug']
        for neighbor in G.neighbors(drug):
            known_edges.append({
                'from': drug,
                'to': neighbor,
                'type': 'known',
                'drug_name': G.nodes[drug].get('label', drug),
                'disease_name': G.nodes[neighbor].get('label', neighbor)
            })
            known_nodes.add(drug)
            known_nodes.add(neighbor)

    # Build node set used in visualization (known + predicted)
    all_nodes = set(known_nodes)
    for pred in unique_preds:
        all_nodes.add(pred['drug'])
        all_nodes.add(pred['disease'])

    # Predicted edges for those drugs
    pred_edges = []
    for pred in unique_preds:
        pred_edges.append({
            'from': pred['drug'],
            'to': pred['disease'],
            'score': pred['score'],
            'type': 'predicted',
            'drug_name': pred['drug_name'],
            'disease_name': pred['disease_name']
        })

    # Graph nodes for viz
    graph_nodes = [
        {
            'id': n,
            'label': G.nodes[n].get('label', n),
            'group': G.nodes[n].get('type', 'unknown')
        }
        for n in all_nodes
    ]

    # Stats
    stats = {
        'num_drugs': sum(1 for n in graph_nodes if n["group"] == "drug"),
        'num_diseases': sum(1 for n in graph_nodes if n["group"] == "disease"),
        'num_edges': len(known_edges) + len(pred_edges),
        'model_type': model_type,
    }

    # Public (JSON-serializable) response
    public_response = {
        'graph': {
            'nodes': graph_nodes,
            'edges': known_edges + pred_edges,
        },
        'predictions': unique_preds,
        'stats': stats
    }

    # Ensure all numpy scalar types are converted to native Python via json round-trip
    try:
        safe_public = json.loads(json.dumps(public_response, default=to_serializable))
    except TypeError as e:
        print("Warning: fallback serialization conversion failed:", e)
        # As a last resort convert arrays/scalars manually
        def manual_convert(obj):
            if isinstance(obj, dict):
                return {k: manual_convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [manual_convert(v) for v in obj]
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        safe_public = manual_convert(public_response)

    # Build final return payload
    # - 'graph_obj' and 'predictions_list' are provided for internal use (e.g. lifespan caching)
    # - 'response' is JSON-safe and should be used when sending HTTP responses
    result = {
        'graph_obj': G,                 # internal: keep original Graph object
        'predictions_list': preds,      # internal: full candidate list
        'response': safe_public         # public: JSON-serializable payload
    }

    return result


if __name__ == '__main__':
    out = run_predictions(model_type='ensemble', top_k=10)
    print("\nTop predictions (public):")
    for i, pred in enumerate(out['response']['predictions'], 1):
        print(f"{i}. {pred['drug_name']} -> {pred['disease_name']}: {pred['score']:.4f}")
