import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from database import save_result, get_all_results
import os

# Download stopwords (only first time)
nltk.download('stopwords')

app = FastAPI()

# -----------------------------
# Extract text from PDF (from file path)
# -----------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/match")
async def match_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    # Ensure resumes folder exists
    os.makedirs("resumes", exist_ok=True)

    # Save uploaded file
    file_location = f"resumes/{resume.filename}"
    with open(file_location, "wb") as f:
        f.write(await resume.read())

    # Extract text from saved file
    resume_text = extract_text_from_pdf(file_location)
    cleaned_resume = clean_text(resume_text)

    # Clean job description
    cleaned_jd = clean_text(job_description)

    # Apply TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

    # Calculate similarity
    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    match_percentage = round(similarity_score * 100, 2)

    status = "Shortlisted" if match_percentage >= 70 else "Rejected"

    # Save result into database
    save_result(resume.filename, match_percentage, status)

    return {
        "resume_name": resume.filename,
        "match_score": match_percentage,
        "status": status
    }

# -----------------------------
# View Results
# -----------------------------
@app.get("/results")
def view_results():
    results = get_all_results()

    formatted_results = []
    for row in results:
        formatted_results.append({
            "id": row[0],
            "resume_name": row[1],
            "match_score": row[2],
            "status": row[3]
        })

    return formatted_results

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
