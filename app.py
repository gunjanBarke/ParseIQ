import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import docx2txt
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime


# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --------- Text Extraction Functions ----------
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text_from_docx(file):
    temp_path = tempfile.mktemp(suffix=".docx")
    with open(temp_path, "wb") as f:
        f.write(file.read())
    return docx2txt.process(temp_path)

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# ---------- Google Sheets Integration ----------
def connect_to_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("resume_score").sheet1  # Replace with your Google Sheet name
    return sheet

def save_results_to_gsheet(data):
    sheet = connect_to_gsheet()
    for row in data:
        sheet.append_row([
            row[0],                          # Resume Name
            round(row[1], 4),                # Similarity Score
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
        ])

# ---------- Resume Ranking ----------
def rank_resumes(job_description, resume_texts):
    job_embed = model.encode(job_description, convert_to_tensor=True)
    rankings = []
    for name, text in resume_texts.items():
        resume_embed = model.encode(text, convert_to_tensor=True)
        score = util.cos_sim(job_embed, resume_embed).item()
        rankings.append((name, score))
    return sorted(rankings, key=lambda x: x[1], reverse=True)

# ---------- Feedback Function ----------
def get_resume_feedback(resume_text, job_description): 
    jd_keywords = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())

    missing = jd_keywords - resume_words
    match_count = len(jd_keywords & resume_words)
    total = len(jd_keywords)

    match_percent = (match_count / total) * 100 if total > 0 else 0

    lines = []
    lines.append("✅ Resume matches about " + str(round(match_percent, 1)) + "% of the job description keywords.")

    if missing:
        lines.append("🔍 Missing keywords (skills or terms): " + ', '.join(list(missing)[:20]) + '.')
    else:
        lines.append("🎯 All job description keywords are present in the resume!")

    lines.append("")
    lines.append("Suggestions:")
    lines.append("- Add relevant keywords from the job description.")
    lines.append("- Clearly highlight matching skills, tools, and achievements.")
    lines.append("- Use consistent formatting and standard section headers (e.g., Experience, Projects).")
    lines.append("- Keep it concise, clean, and professional.")

    return "\n".join(lines)


# ---------- PDF Generation ----------
def generate_pdf(feedback_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    lines = feedback_text.split("\n")
    y = height - 50
    for line in lines:
        if y < 40:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line)
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer

# ---------- Score Plot ----------
def plot_scores(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df.sort_values(by="Similarity Score", ascending=True)  # lowest at bottom
    bars = ax.barh(df_sorted["Resume Name"], df_sorted["Similarity Score"], color='skyblue')

    ax.set_xlabel("Similarity Score")
    ax.set_title("Resume vs Job Description Matching")
    ax.set_xlim(0, 1)  # scores are between 0 and 1

    # Add score labels on each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f"{width:.2f}", va='center')

    st.pyplot(fig)


# ---------- Streamlit App ----------
st.title("📄 AI Resume Ranker (No LLM Version)")
st.write("Upload resumes and a job description file (or paste manually) to rank candidates and receive keyword-based feedback.")

# Job Description input
jd_file = st.file_uploader("📂 Upload Job Description (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
job_description_text = ""

if jd_file:
    if jd_file.name.endswith(".txt"):
        job_description_text = extract_text_from_txt(jd_file)
    elif jd_file.name.endswith(".pdf"):
        job_description_text = extract_text_from_pdf(jd_file)
    elif jd_file.name.endswith(".docx"):
        job_description_text = extract_text_from_docx(jd_file)
else:
    job_description_text = st.text_area("Or Paste Job Description")

# Resume input
resume_files = st.file_uploader("📁 Upload Resumes (.pdf / .docx)", type=["pdf", "docx"], accept_multiple_files=True)
resume_texts = {}

if resume_files:
    for file in resume_files:
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.name.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            text = ""
        resume_texts[file.name] = text

# Run ranking and feedback
if job_description_text and resume_texts:
    ranked = rank_resumes(job_description_text, resume_texts)
        # ✅ Save to Google Sheets
    save_results_to_gsheet(ranked)
    df = pd.DataFrame(ranked, columns=["Resume Name", "Similarity Score"])
    st.subheader("📊 Resume Ranking Results")
    st.dataframe(df)
    plot_scores(df)

    # Prepare combined feedback PDF
    combined_feedback_pdf = io.BytesIO()
    combined_canvas = canvas.Canvas(combined_feedback_pdf, pagesize=letter)
    width, height = letter
    y_start = height - 50

    for name, score in ranked:
        st.subheader(f"📄 Preview: {name}")
        with st.expander("Click to view resume content"):
            st.write(resume_texts[name][:3000])

        with st.spinner("Analyzing resume..."):
            feedback = get_resume_feedback(resume_texts[name], job_description_text)
            st.info(feedback)

            # TXT Feedback
            st.download_button(
                "📥 Download Feedback as TXT",
                feedback,
                file_name=f"{name}_feedback.txt"
            )

            # PDF Feedback for individual download
            pdf_buffer = generate_pdf(feedback)
            st.download_button(
                label="📄 Download Feedback as PDF",
                data=pdf_buffer,
                file_name=f"{name}_feedback.pdf",
                mime="application/pdf"
            )

            # Add to combined PDF
            lines = feedback.split("\n")
            y = y_start
            combined_canvas.setFont("Helvetica-Bold", 12)
            combined_canvas.drawString(50, y, f"Feedback for {name}")
            y -= 20
            combined_canvas.setFont("Helvetica", 11)
            for line in lines:
                if y < 40:
                    combined_canvas.showPage()
                    y = y_start
                combined_canvas.drawString(50, y, line)
                y -= 15
            combined_canvas.showPage()

    # Finalize combined PDF
    combined_canvas.save()
    combined_feedback_pdf.seek(0)

    st.download_button(
        label="📄 Download All Feedbacks as One PDF",
        data=combined_feedback_pdf,
        file_name="combined_resume_feedback.pdf",
        mime="application/pdf"
    )

else:
    st.info("Please upload a job description and at least one resume.")

