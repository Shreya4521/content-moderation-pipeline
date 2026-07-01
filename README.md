# 🛡️ Automated Content Moderation & Compliance Pipeline

An end-to-end pipeline that moderates user-generated text using a **machine learning toxicity classifier** combined with **rule-based compliance checks** (PII leaks, spam, profanity), and logs every decision to an **audit trail** for accountability — similar to how platforms like Reddit, Twitter, or e-commerce review systems handle content moderation at scale.

## Why this project

Most fresher ML projects stop at "train a model, show accuracy." This one goes further by simulating a **real production moderation system**:

- ML model for toxicity detection
- Deterministic rule engine for things ML often misses (PII, spam links)
- Full audit logging (a compliance requirement in real systems — GDPR/DPDP style traceability)
- A working dashboard to demo it live

### Features

- **Single Text Moderation & Privacy Sanitization** — Paste any text to get an instant decision. If personal records are leaked, the system automatically sanitizes the data stream rather than flat-blocking the user.
- **Batch Moderation** — Upload a CSV of comments/reviews to process and analyze content compliance in bulk.
- **Compliance Dashboard** — View live metric counters (Approved vs. Blocked), monitor chart trends, and export structured compliance audit reports.
- **Hybrid Compliance Engine** — Combines a machine learning toxicity classifier with deterministic regex rule modules for multi-layered string validation.

## Tech Stack

| Layer         | Tool                                                                           |
| ------------- | ------------------------------------------------------------------------------ |
| ML Model      | scikit-learn (TF-IDF + Logistic Regression)                                    |
| Rule Engine   | Python `re` (regex)                                                            |
| Dashboard     | Streamlit                                                                      |
| Audit Storage | SQLite                                                                         |
| Data          | Sample dataset included (swap in Kaggle Jigsaw dataset for production quality) |

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

5. **Ingestion:** Text enters the system through the interactive Streamlit dashboard interface or via bulk batch CSV data uploads.
6. **ML Classification:** The text payload runs through the machine learning model vectorizer, generating a toxicity prediction label accompanied by an explicit numerical confidence score.
7. **Deterministic Evaluation:** Simultaneously, a rule-based engine evaluates the string for compliance flags including profanity content, blacklisted keywords, and sensitive PII markers.
8. **Smart Privacy Routing:** If isolated PII data (emails or phone numbers) is caught in an otherwise safe string, a dedicated privacy layer activates **APPROVED** routing and handles data masking protocols instead of binary message blocking.
9. **Traceable Audit Trails:** Every single algorithmic trigger score, classification feature weight, and final routing choice is immediately written to a local SQLite compliance logging database.

## Possible extensions (good for a resume bullet or viva questions)

- Swap Logistic Regression for a fine-tuned transformer (e.g. DistilBERT) for higher accuracy
- Add multi-label classification (hate speech, threats, spam as separate categories instead of one binary flag)
- Add a human-review queue for borderline confidence scores instead of auto-blocking
- Deploy on Streamlit Community Cloud or Hugging Face Spaces for a live demo link

## License

MIT — free to use and modify.
