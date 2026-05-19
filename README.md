# Content Moderation System

A deep learning REST API that classifies free-form text as **clean**, **flagged**, or **toxic** in real time. Built with FastAPI and a GloVe-powered stacked LSTM, trained on the Jigsaw Toxic Comment Classification dataset.

---

## Overview

The system accepts raw text via HTTP and returns a predicted label with confidence scores across all three classes. It is designed for integration into platforms that need automated content screening — comment sections, chat systems, or user-generated content pipelines.

**Classes:**

| Label | Meaning |
|-------|---------|
| `clean` | Acceptable, non-toxic content |
| `flagged` | Borderline or potentially problematic content |
| `toxic` | Abusive, threatening, or severely harmful content |

---

## Model Architecture

### Final Model: GloVe + Stacked LSTM Funnel (`model_v3_glove_lstm_funnel.keras`)

```
Input sequence (max length: 200 tokens)
    ↓
Embedding Layer — GloVe 840B, 300d (vocab: 20,000 words)
    ↓
LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM (32 units, return_sequences=False)
    ↓
Dropout (0.3)
    ↓
Dense (3 units, softmax)
    ↓
Output: [P(clean), P(flagged), P(toxic)]
```

**Pre-trained embeddings:** Stanford GloVe trained on 840B Common Crawl tokens, 300-dimensional vectors. Embeddings were frozen during initial training and then fine-tuned at a lower learning rate (1e-5).

### Architecture Progression

The model evolved through several iterations before reaching the final version:

1. Baseline LSTM (64 units)
2. LSTM with class-balanced weights
3. LSTM with random oversampling
4. LSTM + Dropout
5. LSTM + Dropout + L2 regularization
6. Stacked LSTM (2 layers)
7. **Stacked LSTM Funnel** (128 → 64 → 32)
8. **+ GloVe embeddings** ← deployed

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Batch size | 64 |
| Train/val split | 80 / 20 |
| Class weights | Balanced |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Early stopping | patience=6, monitors val_loss |
| LR scheduler | ReduceLROnPlateau, factor=0.5, patience=4, min_lr=1e-6 |

### Label Derivation (from Jigsaw columns)

- **Clean (0):** `toxic == 0`
- **Toxic (2):** `severe_toxic == 1` OR `threat == 1` OR `identity_hate == 1`
- **Flagged (1):** All other toxic cases

### Classification Thresholds

```
P(toxic)   >= 0.3  →  "toxic"
P(flagged) >= 0.4  →  "flagged"   (evaluated only if not toxic)
otherwise          →  "clean"
```

---

## Text Preprocessing Pipeline

Applied at inference time in `model.py`:

1. Lowercase conversion
2. Remove all non-alphabetic characters (`[^a-z\s]`)
3. Word tokenization via NLTK
4. Convert tokens to integer sequences using the fitted tokenizer (vocab size: 20,000)
5. Pad / truncate to 200 tokens (post-padding, post-truncation)

---

## Tech Stack

| Layer | Library |
|-------|---------|
| API framework | FastAPI |
| ASGI server | Uvicorn |
| Deep learning | TensorFlow / Keras |
| NLP | NLTK |
| Numerics | NumPy |
| Serialization | Joblib |
| Validation | Pydantic |

---

## Project Structure

```
Content-Management-System/
├── main.py                              # FastAPI application (2 endpoints)
├── model.py                             # Preprocessing + inference logic
├── Content_Moderation_System.ipynb      # Training and experimentation notebook
├── requirements.txt                     # Python dependencies
├── artifacts/
│   ├── model_v3_glove_lstm_funnel.keras # Trained model (27 MB)
│   └── tokenizer.joblib                 # Fitted Keras tokenizer (9.2 MB)
└── .gitignore
```

---

## Setup & Installation

**Prerequisites:** Python 3.11+

```bash
# 1. Clone the repository
git clone <repo-url>
cd Content-Management-System

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK tokenizer data
python -m nltk.downloader punkt punkt_tab
```

---

## Running the API

```bash
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`.  
Interactive API docs (Swagger UI): `http://127.0.0.1:8000/docs`

---

## API Reference

### `GET /health`

Liveness check.

**Response:**
```json
{
  "status": "Doing perfectly fine. Thanks for asking"
}
```

---

### `POST /moderate`

Classify a piece of text.

**Request body:**

```json
{
  "text": "string"
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `text` | string | Yes | Must be non-empty after stripping whitespace |

**Response body:**

```json
{
  "text": "string",
  "label": "clean | flagged | toxic",
  "confidence": 0.0000,
  "scores": {
    "clean": 0.0000,
    "flagged": 0.0000,
    "toxic": 0.0000
  }
}
```

| Field | Description |
|-------|-------------|
| `text` | Original input text |
| `label` | Predicted class |
| `confidence` | Probability of the predicted class (4 decimal places) |
| `scores` | Full probability distribution across all three classes |

**Error responses:**

| Status | Cause |
|--------|-------|
| 422 | Empty or missing `text` field |
| 500 | Internal prediction error |

---

## Example

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/moderate \
     -H "Content-Type: application/json" \
     -d '{"text": "Absolute garbage place, the owner is a lying piece of trash"}'
```

**Response:**
```json
{
  "text": "Absolute garbage place, the owner is a lying piece of trash",
  "label": "flagged",
  "confidence": 0.9341,
  "scores": {
    "clean": 0.0298,
    "flagged": 0.9341,
    "toxic": 0.0361
  }
}
```

---

## Model Artifacts

| File | Size | Purpose |
|------|------|---------|
| `artifacts/model_v3_glove_lstm_funnel.keras` | 27 MB | Trained Keras model (GloVe + stacked LSTM) |
| `artifacts/tokenizer.joblib` | 9.2 MB | Fitted tokenizer — maps words to integer indices |

> **Note:** Artifacts are committed to the repository. No external download step is required.

---

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) — a large-scale dataset of Wikipedia talk page comments labeled for toxicity across six categories.
