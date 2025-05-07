# 📄 AI Resume Ranker 

This Streamlit app compares uploaded resumes to a job description using semantic similarity (Sentence Transformers), ranks them, and provides keyword-based feedback.

## 🚀 Features

- Upload and parse PDF/DOCX resumes
- Input job description via file or text
- Rank resumes using semantic similarity
- Generate feedback in TXT and PDF
- Upload scores to Google Sheets

## 📦 Setup

```bash
pip install -r requirements.txt
streamlit run app.py
