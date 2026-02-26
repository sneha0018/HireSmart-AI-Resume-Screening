import pdfplumber
import re
import nltk
import os
import boto3
import io
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from botocore.config import Config

# -----------------------------
# NLTK Setup
# -----------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="supersecretkey123")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# PostgreSQL Config (RDS)
# -----------------------------
DB_HOST = "hiresmart-db.c9s4we6egseb.eu-north-1.rds.amazonaws.com"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "12345678"
DB_PORT = "5432"

# -----------------------------
# S3 Config
# -----------------------------
s3 = boto3.client(
    "s3",
    region_name="eu-north-1",
    config=Config(signature_version="s3v4"),
)
S3_BUCKET = "hiresmart-resumes-sneha"

# -----------------------------
# Password Hashing
# -----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -----------------------------
# DB Connection
# -----------------------------
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        cursor_factory=RealDictCursor,
    )

# -----------------------------
# Create Tables
# -----------------------------
def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id SERIAL PRIMARY KEY,
            resume_name TEXT,
            match_score FLOAT,
            status TEXT,
            s3_key TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

create_tables()

# -----------------------------
# Create Default Admin
# -----------------------------
def create_admin():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = %s", ("admin",))
    existing = cursor.fetchone()

    if not existing:
        hashed_password = pwd_context.hash("hiresmart123")
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            ("admin", hashed_password),
        )
        conn.commit()

    conn.close()

create_admin()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# LOGIN
# -----------------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    conn.close()

    if user and pwd_context.verify(password, user["password"]):
        request.session["user"] = username
        return RedirectResponse(url="/dashboard", status_code=302)

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password"},
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)

# -----------------------------
# PUBLIC PAGES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/match", response_class=HTMLResponse)
def match_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, title FROM jobs ORDER BY created_at DESC")
    jobs = cursor.fetchall()

    conn.close()

    return templates.TemplateResponse(
        "match.html",
        {"request": request, "jobs": jobs}
    )
# -----------------------------
# MATCH LOGIC
# -----------------------------
@app.post("/match", response_class=HTMLResponse)
async def match_resume(
    request: Request,
    file: UploadFile = File(...),
    job_id: int = Form(...)
):

    file_bytes = await file.read()

    # Upload to S3
    s3_key = f"resumes/{file.filename}"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=file_bytes
    )

    # Extract PDF text safely
    resume_text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    resume_text += text
    except:
        return templates.TemplateResponse(
            "match.html",
            {"request": request, "error": "Invalid PDF file."},
        )
   # Get job description from DB
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT description FROM jobs WHERE id = %s", (job_id,))
    job = cursor.fetchone()

    if not job:
        conn.close()
        return templates.TemplateResponse(
            "match.html",
            {"request": request, "error": "Job not found."},
        )

    job_description = job["description"]
    conn.close()

    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_clean, job_clean])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    tfidf_score = round(similarity * 100, 2)

    job_keywords = set(job_clean.split())
    resume_keywords = set(resume_clean.split())
    matched_skills = job_keywords.intersection(resume_keywords)

    skill_score = round(
        (len(matched_skills) / len(job_keywords)) * 100, 2
    ) if job_keywords else 0

    final_score = float(round((0.7 * tfidf_score) + (0.3 * skill_score), 2))
    status = "Selected" if final_score >= 60 else "Rejected"

    # Save to PostgreSQL
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO results (resume_name, match_score, status, s3_key, job_id) VALUES (%s, %s, %s, %s, %s)",
        (file.filename, final_score, status, s3_key, job_id),
    )
    conn.commit()
    conn.close()

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "score": final_score,
            "status": status,
            "matched_skills": list(matched_skills),
        },
    )

# -----------------------------
# DASHBOARD
# -----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):

    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
       SELECT r.id, r.resume_name, r.match_score, r.status, j.title
       FROM results r
       JOIN jobs j ON r.job_id = j.id
       ORDER BY r.match_score DESC
    """)
    results = cursor.fetchall()

    total = len(results)
    selected = len([r for r in results if r["status"] == "Selected"])
    rejected = total - selected
    avg_score = round(sum([r["match_score"] for r in results]) / total, 2) if total > 0 else 0

    conn.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "results": results,
            "total": total,
            "selected": selected,
            "rejected": rejected,
            "avg_score": avg_score,
        },
    )
#-------------------------------
# ANALYTICS
#-------------------------------
@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request):

    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM results")
    results = cursor.fetchall()

    total = len(results)
    selected = len([r for r in results if r["status"] == "Selected"])
    rejected = total - selected

    conn.close()

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "total": total,
            "selected": selected,
            "rejected": rejected,
        },
    )
# -----------------------------
# DOWNLOAD FROM S3
# -----------------------------
@app.get("/download/{resume_id}")
def download_resume(request: Request, resume_id: int):

    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT s3_key FROM results WHERE id = %s", (resume_id,))
    result = cursor.fetchone()
    conn.close()

    if not result:
        return {"error": "Resume not found."}

    s3_key = result["s3_key"]

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": s3_key,
        },
        ExpiresIn=300,
    )

    return RedirectResponse(url)
# -----------------------------
# CREATE JOB (ADMIN)
# -----------------------------
@app.get("/create-job", response_class=HTMLResponse)
def create_job_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse("create_job.html", {"request": request})


@app.post("/create-job")
async def create_job(
    request: Request,
    title: str = Form(...),
    description: str = Form(...)
):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO jobs (title, description) VALUES (%s, %s)",
        (title, description),
    )

    conn.commit()
    conn.close()

    return RedirectResponse(url="/dashboard", status_code=302)
