# 📄 AI Resume Screening & Candidate Shortlisting

> Intelligent resume screening powered by NLP and Sentence-BERT. Automatically parse, analyze, and rank candidates with explainable scoring.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square&logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-Sentence--BERT-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## ✨ Features

- **📤 Multi-format Resume Upload** — Support for PDF, DOCX, and TXT files with robust parsing
- **🔍 Smart Skill Extraction** — Taxonomy-based skill recognition with intelligent aliases (e.g., ML → Machine Learning)
- **🧠 Semantic Matching** — Powered by Sentence-BERT (`all-MiniLM-L6-v2`) with intelligent chunking for long documents
- **⚖️ Hybrid Scoring** — Combines semantic similarity, skill matches, experience, education, and bonus factors with adjustable weights
- **🔍 Explainable Results** — View matched/missing skills, component-level score breakdowns, and confidence metrics
- **🙈 Blind Screening Mode** — Automatically mask emails, phone numbers, URLs, and pronouns for unbiased evaluation
- **📊 Export & Analytics** — Generate CSV reports for further analysis and integration with recruitment pipelines

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/Chdrryy-nndnii211/ai-resume-screening.git
cd ai-resume-screening

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Docker

```bash
docker build -t resume-screener .
docker run -p 8501:8501 resume-screener
```

## 📋 Project Structure

```
ai-resume-screening/
├── app.py                          # Streamlit web interface
├── screening.py                    # Core ML engine & scoring logic
├── notebooks/
│   └── resume_screening.ipynb      # Training & evaluation notebook (Colab-ready)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
└── README.md
```

## 📊 Performance Benchmarks

| Method | P@10 | NDCG@10 |
|--------|------|---------|
| TF-IDF Baseline | — | — |
| Sentence-BERT Only | — | — |
| Hybrid (SBERT + Skills) | — | — |

*Evaluated on Kaggle Resume Dataset (24 job categories, ~2,400 resumes)*

## 📦 Dataset

This project uses the [Resume Dataset from LiveCareer](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) available on Kaggle:
- **24 job categories**
- **~2,400 resumes**
- Download via the Jupyter notebook

## 🎯 How It Works

1. **Resume Parsing** — Extract text from PDF/DOCX/TXT files
2. **Skill Matching** — Identify skills against job taxonomy with alias support
3. **Semantic Scoring** — Compare resume embeddings with job description using Sentence-BERT
4. **Hybrid Ranking** — Combine multiple signals (semantics, skills, experience) into a unified score
5. **Explainability** — Break down the score and highlight matched/missing qualifications
6. **Optional Blind Review** — Remove identifying information for bias-free screening

## ⚠️ Limitations & Future Work

- **OCR** — Scanned PDFs currently unsupported; add `pytesseract` for OCR capability
- **Name Masking** — Blind mode masks contact info and pronouns but retains names; extend with spaCy PERSON NER
- **Ground Truth** — Currently uses category labels as relevance proxy; recruiter annotations would improve accuracy
- **Human-in-the-Loop** — Tool is a decision support system; final hiring decisions remain with recruiters

## 🌐 Live Demo

[Try the live demo on Streamlit Cloud](https://ai-resume-screening-21.streamlit.app)

## 📚 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **ML/NLP** | Sentence-BERT, scikit-learn |
| **Text Processing** | python-docx, pdfplumber, nltk |
| **Container** | Docker |
| **Language** | Python 3.8+ |

## 📝 Usage Example

```python
from screening import ResumeScreener

screener = ResumeScreener(model_name='all-MiniLM-L6-v2')

# Score a resume against a job description
score = screener.score_resume(
    resume_text="...",
    job_description="...",
    weights={
        'semantic': 0.4,
        'skills': 0.3,
        'experience': 0.2,
        'education': 0.1
    }
)

print(score)  # Returns detailed breakdown
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🙏 Acknowledgments

- [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Sneha Anabhawal
- [Sentence-BERT](https://www.sbert.net/) by UKP Lab
- [Streamlit](https://streamlit.io/) for the incredible framework

## 📞 Contact & Support

Have questions or suggestions? Feel free to:
- Open an [issue](https://github.com/Chdrryy-nndnii211/ai-resume-screening/issues)
- Start a [discussion](https://github.com/Chdrryy-nndnii211/ai-resume-screening/discussions)
- Reach out directly

---
## Snapshots
<img width="1366" height="768" alt="Screenshot (139)" src="https://github.com/user-attachments/assets/4b0ecb90-bdd4-4344-9470-299796b5dae9" />
<img width="1366" height="768" alt="Screenshot (140)" src="https://github.com/user-attachments/assets/26f794e6-bed8-4f76-a889-0e6a2dd70984" />
<img width="1366" height="768" alt="Screenshot (141)" src="https://github.com/user-attachments/assets/5a173e81-c376-4d91-a3af-6b6c0040c07a" />
<img width="1366" height="768" alt="Screenshot (142)" src="https://github.com/user-attachments/assets/5fd0e775-2032-4b17-af65-0675e0a684a2" />








**Made with ❤️ by [Chdrryy-nndnii211](https://github.com/Chdrryy-nndnii211)**
