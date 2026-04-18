#  Sentiment Analysis Pipeline , DistilBERT + FastAPI

**Custom-trained DistilBERT model • Evaluation & Error Analysis • FastAPI • Docker • Render Deployment**

A production-style NLP pipeline for binary sentiment classification, fine-tuned on IMDB reviews with full evaluation analysis. Built end-to-end with FastAPI, Docker, and deployed to Render.com.

**[▶ Live Web App](https://nlp-sentiment-pipeline.onrender.com)** · **[API Docs](https://nlp-sentiment-pipeline.onrender.com/docs)** · **[Model on HuggingFace](https://huggingface.co/aliabbi/sentiment-distilbert)**

---

## Results — Evaluation on IMDB Test Set

Evaluated on 1,000 random samples from the IMDB test set.

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 86.70% |
| F1 (weighted) | 86.69% |
| Precision   | 86.72% |
| Recall      | 86.70% |

### Per-Class Performance

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.8596    | 0.8848 | 0.8720   | 512     |
| Positive | 0.8753    | 0.8484 | 0.8616   | 488     |

The model shows slight negative-class bias, it catches negative reviews more reliably (recall 0.88) than positive ones (recall 0.85).

---

## Error Analysis

Of 1,000 test samples, 133 were misclassified. Breaking down by confidence:

| Confidence Bucket | Count | What It Tells Us |
|---|---|---|
| >0.9 (high)       | 52    | Model was **wrong AND confident** — the worst failure mode |
| 0.7–0.9 (mid)     | 56    | Moderate errors, often ambiguous reviews |
| <0.7 (low)        | 25    | Model signaled uncertainty — acceptable errors |

**False Positives** (negative review → predicted positive): 59
**False Negatives** (positive review → predicted negative): 74

### Common Failure Patterns

Inspection of high-confidence misclassifications revealed three patterns:

1. **Surface-level sentiment bias** — Reviews containing many positive words but overall negative meaning get misclassified. Example: a negative review describing a "beautiful, appealing" actress in a "terminal illness" film was predicted positive at 97.2% confidence.

2. **Context-dependent sentiment** — "Heartbreaking" in a critic review about Lost was praise, but the model read it as negative. Positive reviews with emotionally heavy language suffer this.

3. **Review length** — The 256-token truncation limit cuts off key context in longer reviews, impacting predictions on detailed criticism.

Full misclassified examples: `evaluation_outputs/misclassified_examples.csv`

---

## Known Limitations

- **Overconfidence on errors** — 39% of errors occurred at >0.9 confidence. The model needs confidence calibration (e.g., temperature scaling) for production use.
- **Binary only** — Trained for positive/negative; real-world sentiment has neutral/mixed cases.
- **Domain shift** — Trained on movie reviews; performance on product reviews, tweets, or formal text would require domain-specific fine-tuning.
- **Short context** — 256-token limit truncates long reviews, losing key context.

### Improvements With More Time

1. **Confidence calibration** — Apply temperature scaling so predicted probabilities reflect actual accuracy
2. **Longer context** — Use Longformer or chunk-and-aggregate for full review processing
3. **Multi-class** — Extend to 5-star ratings or neutral/mixed categories
4. **Adversarial examples** — Train on sarcasm and mixed-sentiment examples to reduce surface-word bias

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| Base model | DistilBERT-base-uncased | 40% smaller than BERT, retains 97% performance |
| Training | HuggingFace Trainer | Standard fine-tuning workflow |
| Evaluation | scikit-learn, matplotlib, seaborn | Classification report + confusion matrix |
| API | FastAPI | Auto-docs, Pydantic validation, async-ready |
| Container | Docker | CPU-only, reproducible |
| Deployment | Render.com | Auto-redeploy on git push |
| Model hosting | HuggingFace Hub | Public model card, versioned weights |

---

## Architecture

```
Training (Kaggle/local)
         ↓
IMDB reviews → clean → tokenize → fine-tune DistilBERT → save model
         ↓
HuggingFace Hub (aliabbi/sentiment-distilbert)
         ↓
FastAPI pulls model → /predict endpoint + HTML UI
         ↓
Docker container → Render.com → Live URL
```

---

## Project Structure

```
nlp-sentiment-pipeline/
│
├── api/
│   └── app.py                      # FastAPI server (JSON API + HTML UI)
│
├── src/
│   ├── data/load_data.py           # IMDB dataset loading
│   ├── utils/text_cleaning.py      # preprocessing
│   └── models/train_distilbert.py  # fine-tuning script
│
├── tests/                          # pytest suite
│
├── evaluation_outputs/
│   ├── confusion_matrix.png
│   └── misclassified_examples.csv
│
├── evaluate.py                     # evaluation + error analysis
├── Dockerfile
├── requirements.txt
├── start.sh
└── README.md
```

---

## Run Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run FastAPI
```bash
uvicorn api.app:app --reload
# → http://localhost:8000       (HTML UI)
# → http://localhost:8000/docs  (API docs)
```

### Run with Docker
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

### Run evaluation
```bash
pip install scikit-learn matplotlib seaborn
python evaluate.py
```

---

## API Usage

### POST `/predict`

**Request:**
```json
{
  "text": "I loved this movie!"
}
```

**Response:**
```json
{
  "label": "positive",
  "probabilities": {
    "negative": 0.02,
    "positive": 0.98
  }
}
```

---

## Testing

```bash
pytest -q
```

Covers data loading, text cleaning, and API responses.

---

## About

Built as part of a portfolio while targeting NLP/GenAI engineering roles in the Netherlands.

**Author:** Ali Aghababaei
**Contact:** [LinkedIn](https://linkedin.com/in/aliaghababaeii) · [Portfolio](https://aghababaeiali.github.io)

**License:** MIT