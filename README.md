# Graph-Based Drug Repurposing Engine Using Graph Machine Learning

## Overview

Drug repurposing provides a cost-effective and time-efficient alternative to conventional drug discovery by identifying new therapeutic applications for existing drugs. This project proposes a graph-based drug repurposing engine that utilizes Graph Machine Learning (GML) techniques to predict novel drug–disease associations.

Biomedical data is modeled as a bipartite graph where drugs and diseases are represented as nodes, and known therapeutic relationships are represented as edges. By leveraging graph neural networks and graph autoencoders, the system learns latent structural and relational patterns within the graph to recommend potential drug repurposing candidates.

---

## Objectives

* To construct a drug–disease interaction graph from curated biomedical datasets
* To generate meaningful node embeddings using graph neural network models
* To identify missing or novel drug–disease associations through link prediction
* To rank predicted associations based on confidence scores

---

## System Architecture

1. **Data Collection and Preprocessing**
   Drug–disease association data is collected from publicly available biomedical sources. The data is cleaned, normalized, and transformed into a graph-compatible format.

2. **Graph Construction**
   A bipartite graph is created with drugs and diseases as nodes and therapeutic relationships as edges.

3. **Feature Engineering**

   * Drug features are derived using molecular fingerprints or learned embeddings.
   * Disease features are generated using pathway-based or ontology-driven encodings.

4. **Graph Machine Learning Models**

   * Graph Convolutional Networks (GCN)
   * GraphSAGE
   * Graph Attention Networks (GAT)
   * Graph Autoencoders (GAE) and Variational Graph Autoencoders (VGAE) for link reconstruction

5. **Ensemble Prediction**
   Predictions from multiple models are combined to improve robustness and generalization.

6. **Evaluation and Visualization**
   Model performance is evaluated using standard metrics, and predicted associations are visualized for analysis.

---

## Models Used

| Model     | Description                                    |
| --------- | ---------------------------------------------- |
| GCN       | Learns global graph representations            |
| GraphSAGE | Supports inductive learning on unseen nodes    |
| GAT       | Assigns attention weights to neighboring nodes |
| GAE       | Reconstructs graph structure                   |
| VGAE      | Probabilistic link prediction                  |

---

## Technology Stack

* **Programming Language:** Python
* **Frameworks and Libraries:**

  * PyTorch
  * PyTorch Geometric
  * NetworkX
  * NumPy
  * Pandas
  * Matplotlib / Seaborn
* **Development Environment:** VS Code, Jupyter Notebook

---

## Evaluation Metrics

* Area Under the Curve (AUC)
* Average Precision (AP)

---

## How to Run

```bash
git clone https://github.com/your-username/drug-repurposing-gml.git
cd drug-repurposing-gml
pip install -r requirements.txt
python main.py
```

---

## Results

The proposed system successfully predicts potential novel drug–disease associations. The ensemble approach enhances prediction stability and demonstrates the effectiveness of graph-based learning techniques in biomedical applications.

---

## Future Work

* Integration of heterogeneous knowledge graphs including genes and proteins
* Extension to temporal graph modeling
* Deployment as a web-based drug recommendation system
* Incorporation of explainability mechanisms for model interpretability

---

## Contributors

Akanksha Shetty
Anjana Rajagopal

---

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
2. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs.
3. Zitnik, M., Agrawal, M., & Leskovec, J. (2018). Modeling Polypharmacy Side Effects with Graph Neural Networks.

