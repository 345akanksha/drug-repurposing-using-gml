from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from ml_engine.predict import run_predictions

# Global cache
G = None
preds = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global G, preds
    data = run_predictions(model_type='ensemble', top_k=100)
    G = data["graph_obj"]
    preds = data["predictions_list"]
    yield
    # Cleanup (optional)

app = FastAPI(
    title="Drug Repurposing GNN API",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Drug Repurposing GNN API",
        "endpoints": {
            "/data": "Get predictions with ?model=ensemble&top_k=20",
            "/drug_neighborhood": "Get neighborhood for a drug with ?drug_id=<drug>"
        }
    }

@app.get("/data")
def get_data(model: str = Query('ensemble'), top_k: int = Query(20)):
    try:
        data = run_predictions(model_type=model, top_k=top_k)
        return JSONResponse(content=data['response'])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/drug_neighborhood")
def drug_neighborhood(drug_id: str = Query(...)):
    global G, preds
    if G is None or preds is None:
        return JSONResponse(status_code=503, content={"error": "Graph not loaded yet."})
    try:
        neighbors = set(G.neighbors(drug_id))
    except Exception:
        return JSONResponse(status_code=404, content={"error": f"Drug {drug_id} not found"})
    
    pred_diseases = set([p['disease'] for p in preds if p['drug'] == drug_id])
    all_neighbors = neighbors | pred_diseases

    nodes = [{"id": drug_id, "label": G.nodes[drug_id].get("label", drug_id), "group": "drug"}] + [
        {"id": n, "label": G.nodes[n].get("label", n), "group": G.nodes[n].get("type", "unknown")}
        for n in all_neighbors
    ]

    edges = []
    for n in neighbors:
        edges.append({
            "from": drug_id,
            "to": n,
            "type": "known",
            "drug_name": G.nodes[drug_id].get("label", drug_id),
            "disease_name": G.nodes[n].get("label", n)
        })
    for n in pred_diseases:
        edges.append({
            "from": drug_id,
            "to": n,
            "type": "predicted",
            "drug_name": G.nodes[drug_id].get("label", drug_id),
            "disease_name": G.nodes[n].get("label", n)
        })
    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
