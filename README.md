# 🛡️ Automated Content Moderation & Compliance Pipeline

An end-to-end pipeline that moderates user-generated text using a **machine learning toxicity classifier** combined with **rule-based compliance checks** (PII leaks, spam, profanity), and logs every decision to an **audit trail** for accountability — similar to how platforms like Reddit, Twitter, or e-commerce review systems handle content moderation at scale.

## Why this project

Most fresher ML projects stop at "train a model, show accuracy." This one goes further by simulating a **real production moderation system**:
- ML model for toxicity detection
- Deterministic rule engine for things ML often misses (PII, spam links)
- Full audit logging (a compliance requirement in real systems — GDPR/DPDP style traceability)
- A working dashboard to demo it live

## Features

- **Single text moderation** — paste any text and get an instant Approve/Block decision
- **Batch moderation** — upload a CSV of comments/reviews and moderate them all at once
- **Compliance dashboard** — view stats (approved vs blocked), full audit log, and export a compliance report
- **Hybrid detection** — ML classifier (TF-IDF + Logistic Regression) + regex-based rules for PII, spam, profanity

## Tech Stack

| Layer | Tool |
|---|---|
| ML Model | scikit-learn (TF-IDF + Logistic Regression) |
| Rule Engine | Python `re` (regex) |
| Dashboard | Streamlit |
| Audit Storage | SQLite |
| Data | Sample dataset included (swap in Kaggle Jigsaw dataset for production quality) |

## Project Structure

```
content-moderation-pipeline/
├── app.py                  # Streamlit dashboard (entry point)
├── requirements.txt
├── data/
│   └── sample_dataset.csv  # small labeled dataset (toxic / safe)
├── src/
│   ├── rules.py            # PII, spam, profanity rule checks
│   ├── train_model.py      # trains and saves the ML model
│   └── database.py         # SQLite audit logging
├── models/                 # generated after training (gitignored)
└── logs/                   # generated audit DB (gitignored)
```

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd content-moderation-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates models/toxicity_model.pkl)
python src/train_model.py

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Using a real-world dataset

The bundled `sample_dataset.csv` is intentionally small (~40 rows) so the repo stays lightweight and the model trains instantly. For a stronger model:

1. Download the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) from Kaggle
2. Place `train.csv` inside `data/`
3. Update `DATA_PATH` in `src/train_model.py` to point to it, and map its label columns to a single binary `label` column
4. Re-run `python src/train_model.py`

## How it works

1. Text comes in through the UI (single input or CSV upload)
2. It's run through the **ML classifier**, producing a toxic/safe prediction with a confidence score
3. It's simultaneously run through the **rule engine**, checking for emails/phone numbers (PII), spam links/keywords, and profanity
4. If either the model flags it as toxic OR any rule fires, the content is marked `BLOCKED`
5. Every decision — flagged or not — is written to a SQLite audit log with a timestamp, so there's a traceable compliance history

## Possible extensions (good for a resume bullet or viva questions)

- Swap Logistic Regression for a fine-tuned transformer (e.g. DistilBERT) for higher accuracy
- Add multi-label classification (hate speech, threats, spam as separate categories instead of one binary flag)
- Add a human-review queue for borderline confidence scores instead of auto-blocking
- Deploy on Streamlit Community Cloud or Hugging Face Spaces for a live demo link

## License

MIT — free to use and modify.
