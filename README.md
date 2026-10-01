# AI-Based Resume Screening and Candidate Shortlisting System

A decision-support tool that parses resumes (PDF/DOCX/TXT), compares them with a job
description using NLP + Sentence-BERT, and produces an explainable, ranked shortlist.

## Features

- Bulk resume upload (PDF, DOCX, TXT)
- Skill extraction using a skill taxonomy with aliases (e.g. "ML" -> "machine learning")
- Semantic matching with Sentence-BERT (`all-MiniLM-L6-v2`), chunked for long resumes
- Hybrid score = semantic + skill match + experience + education + extras (weights adjustable)
- Explainability: matched skills, missing skills, per-component breakdown
- Blind screening mode (masks emails, phones, links, pronouns)
- Export shortlist to CSV

## Project structure

```
.
├── app.py              # Streamlit frontend
├── screening.py        # core engine (parsing, skills, embeddings, scoring)
├── notebooks/
│   └── resume_screening.ipynb   # Colab notebook: data, training, evaluation
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

[Resume Dataset (Livecareer)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) from Kaggle,
24 job categories, ~2,400 resumes. Not included in this repo; download it via the notebook.

## Results

| Method                  | P@10      | NDCG@10   |
| ----------------------- | --------- | --------- |
| TF-IDF baseline         | _fill in_ | _fill in_ |
| SBERT only              | _fill in_ | _fill in_ |
| Hybrid (SBERT + skills) | _fill in_ | _fill in_ |

## Limitations

- Scanned PDFs need OCR (not enabled in the web app).
- Blind mode masks contact info and pronouns but not names (add spaCy PERSON masking to extend).
- Category labels are used as a proxy for relevance; a recruiter-labelled gold set would be better.
- Decision support only; humans make the final hiring decision.

## Live demo

https://ai-resume-screening-21.streamlit.app
