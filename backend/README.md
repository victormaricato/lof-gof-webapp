# LOF-GOF Predictor API (Render)

FastAPI service that wraps the LOF/GOF classifier. It builds ESM2 embeddings for a reference protein sequence, applies a single-point mutation, and scores the variant with the bundled LightGBM model.

## Run locally
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Request example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACDEFGHIKLMNPQRSTVWY","position":3,"alt":"A"}'
```

## Render deployment (web service)
Render YAML is in `render.yaml`. The heavy ESM2 model requires >2 GB RAM; start with a paid plan or scale up if the free tier OOMs.

Steps:
1) Push this folder to a new Git repo (or set `rootDir: backend` in Render).  
2) New Web Service → Environment: Python → Build command: `pip install -r requirements.txt` → Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.  
3) Env vars:
   - `ESM_MODEL_NAME` = `facebook/esm2_t33_650M_UR50D`
   - `CLASSIFIER_PATH` = `model/lgb_esm2.pkl`
   - `TORCH_NUM_THREADS` = `1`
   - `MAX_SEQUENCE_LENGTH` = `1024`
4) Enable auto-deploy.

## API
- `GET /health` – readiness probe and model info.  
- `POST /predict` – body: `{ "sequence": "...", "position": 123, "alt": "A" }`. Response: `{ "label": "LOF|GOF|NEUTRAL", "probabilities": { ... } }`.
