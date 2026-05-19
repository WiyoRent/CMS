import re
import nltk
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize

nltk.download('punkt_tab', quiet = True)


LABELS = ['clean', 'flagged', 'toxic']
MAXLEN = 200

TOKENIZER_PATH = 'artifacts/tokenizer.joblib'
MODEL_PATH = 'artifacts/model_v3_glove_lstm_funnel.keras'

tokenizer = joblib.load(TOKENIZER_PATH)
model = load_model(MODEL_PATH)

TOXIC_THRESHOLD = 0.3
FLAGGED_THRESHOLD = 0.4


def standardize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(
        "[^a-z\s]",
        "",
        text
    )
    tokens = word_tokenize(text)
    return ' '.join(tokens)


def predict(text: str) -> dict:
    standardized_text = standardize_text(text)
    text_sequence = tokenizer.texts_to_sequences([standardized_text])
    padded_text = pad_sequences(text_sequence, maxlen=MAXLEN, truncating = 'post', padding = 'post')
    probs = model.predict(padded_text, verbose=0)[0]

    if probs[2] >= TOXIC_THRESHOLD:
        label_idx = 2
    elif probs[1] >= FLAGGED_THRESHOLD:
        label_idx = 1
    else:
        label_idx = 0

    return {
        'label': LABELS[label_idx],
        'confidence': round(float(probs[label_idx]), 4),
        'scores': {
            label: float(probs[i]) for i, label in enumerate(LABELS)
        }
    }

