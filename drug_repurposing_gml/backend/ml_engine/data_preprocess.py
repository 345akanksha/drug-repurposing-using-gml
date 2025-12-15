import pandas as pd
import networkx as nx
from pathlib import Path
import numpy as np

DATA_PATH = Path('datasets')

def load_csvs(data_path=DATA_PATH):
    """Load drug, disease, and mapping CSV files"""
    drugs_fp = data_path / 'drugsInfo.csv'
    diseases_fp = data_path / 'diseasesInfo.csv'
    mapping_fp = data_path / 'mapping.csv'
    
    if not drugs_fp.exists() or not diseases_fp.exists() or not mapping_fp.exists():
        raise FileNotFoundError(
            f"Place your CSV files in {data_path} as 'drugsInfo.csv', "
            "'diseasesInfo.csv' and 'mapping.csv'."
        )
    
    drugs = pd.read_csv(drugs_fp)
    diseases = pd.read_csv(diseases_fp)
    mapping = pd.read_csv(mapping_fp)
    
    return drugs, diseases, mapping

def build_graph(drugs=None, diseases=None, mapping=None):
    """
    Build a bipartite graph from drug and disease data.
    
    Returns:
        tuple: (G, num_drugs, num_diseases) where G is NetworkX graph
    """
    if drugs is None or diseases is None or mapping is None:
        drugs, diseases, mapping = load_csvs()
    
    G = nx.Graph()
    
    # Add drug nodes
    for _, row in drugs.iterrows():
        node_id = str(row['DrugID'])
        G.add_node(
            node_id, 
            type='drug', 
            label=row.get('DrugName', node_id)
        )
    
    # Add disease nodes
    for _, row in diseases.iterrows():
        node_id = str(row['DiseaseID'])
        G.add_node(
            node_id, 
            type='disease', 
            label=row.get('DiseaseName', node_id)
        )
    
    # Add edges from mapping
    for _, row in mapping.iterrows():
        drug_id = str(row['DrugID'])
        disease_id = str(row['DiseaseID'])
        if drug_id in G.nodes and disease_id in G.nodes:
            G.add_edge(drug_id, disease_id)
    
    # Compute node features: [degree, is_drug, is_disease]
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        degree = G.degree(node)
        is_drug = 1.0 if node_type == 'drug' else 0.0
        is_disease = 1.0 if node_type == 'disease' else 0.0
        G.nodes[node]['features'] = np.array([degree, is_drug, is_disease], dtype=np.float32)
    
    # Count nodes by type
    num_drugs = sum(1 for _, attr in G.nodes(data=True) if attr.get('type') == 'drug')
    num_diseases = sum(1 for _, attr in G.nodes(data=True) if attr.get('type') == 'disease')
    
    print(f"Built graph: {num_drugs} drugs, {num_diseases} diseases, {G.number_of_edges()} edges")
    
    return G, num_drugs, num_diseases

if __name__ == '__main__':
    # Test the graph building
    G, n_drugs, n_diseases = build_graph()
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    print(f"Drugs: {n_drugs}, Diseases: {n_diseases}")