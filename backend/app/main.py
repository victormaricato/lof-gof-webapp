import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from transformers import AutoTokenizer, EsmModel

LABELS = ["NEUTRAL", "LOF", "GOF"]
MODEL_NAME = os.getenv("ESM_MODEL_NAME", "facebook/esm2_t33_650M_UR50D")
CLASSIFIER_PATH = os.getenv(
    "CLASSIFIER_PATH",
    str(Path(__file__).resolve().parent.parent / "model" / "lgb_esm2.pkl"),
)
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "1024"))

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))


class PredictRequest(BaseModel):
    sequence: constr(strip_whitespace=True, min_length=1)
    position: int = Field(..., gt=0, description="1-based amino acid position")
    alt: constr(strip_whitespace=True, min_length=1, max_length=1)


class PredictResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]


@lru_cache(maxsize=1)
def get_classifier():
    path = Path(CLASSIFIER_PATH).resolve()
    if not path.exists():
        raise RuntimeError(f"Classifier not found at {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_esm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def mean_embedding(sequence: str) -> np.ndarray:
    tokenizer, model, device = get_esm()
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding.astype(np.float32)


def apply_mutation(sequence: str, position: int, alt: str) -> str:
    if position > len(sequence):
        raise ValueError("Position is greater than sequence length.")
    seq_list = list(sequence)
    seq_list[position - 1] = alt
    return "".join(seq_list)


def predict_variant(sequence: str, position: int, alt: str):
    ref_embedding = mean_embedding(sequence)
    alt_sequence = apply_mutation(sequence, position, alt)
    alt_embedding = mean_embedding(alt_sequence)
    features = np.concatenate([ref_embedding, alt_embedding])
    clf = get_classifier()
    proba = clf.predict_proba([features])[0]
    pred_idx = int(np.argmax(proba))
    return LABELS[pred_idx], {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}


app = FastAPI(title="LOF-GOF Predictor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    tokenizer, _, _ = get_esm()
    return {"status": "ok", "model": MODEL_NAME, "vocab": len(tokenizer)}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    sequence = payload.sequence.strip().upper()
    alt = payload.alt.strip().upper()
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"Sequence too long; max {MAX_SEQUENCE_LENGTH} aa."
        )
    if alt not in set("ACDEFGHIKLMNPQRSTVWY"):
        raise HTTPException(status_code=400, detail="ALT must be a valid amino acid code.")
    try:
        label, probabilities = predict_variant(sequence, payload.position, alt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(label=label, probabilities=probabilities)
