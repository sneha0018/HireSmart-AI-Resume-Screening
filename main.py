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
from fastapi.responses import StreamingResponse
import csv
import io

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

    search = request.query_params.get("search")
    sort = request.query_params.get("sort")
    selected_job = request.query_params.get("job_id")

    conn = get_connection()
    cursor = conn.cursor()

    # Get all jobs for dropdown
    cursor.execute("SELECT id, title FROM jobs ORDER BY created_at DESC")
    jobs = cursor.fetchall()

    # Base query
    query = """
        SELECT r.id, r.resume_name, r.match_score, r.status,
               r.notes, j.title
        FROM results r
        LEFT JOIN jobs j ON r.job_id = j.id
    """

    conditions = []
    values = []

    # Search filter
    if search:
        conditions.append("(r.resume_name ILIKE %s OR j.title ILIKE %s)")
        values.extend([f"%{search}%", f"%{search}%"])

    # Job filter
    if selected_job:
        conditions.append("r.job_id = %s")
        values.append(selected_job)

    # Apply conditions
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # Sorting
    if sort == "score":
        query += " ORDER BY r.match_score DESC"
    else:
        query += " ORDER BY r.id DESC"

    # Execute
    cursor.execute(query, values)
    results = cursor.fetchall()

    conn.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "results": results,
            "jobs": jobs,
            "selected_job": selected_job
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
# UPDATE CANDIDATE
# -----------------------------


@app.post("/update-candidate/{candidate_id}")
async def update_candidate(
    request: Request,
    candidate_id: int,
    status: str = Form(...),
    notes: str = Form(None)
):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE results
        SET status = %s,
            notes = %s
        WHERE id = %s
    """, (status, notes, candidate_id))

    conn.commit()
    conn.close()

    return RedirectResponse(url="/dashboard", status_code=302)
# -----------------------------
# DELETE RESUME
# -----------------------------
@app.get("/delete/{result_id}")
def delete_resume(result_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    # Get s3 key first
    cursor.execute("SELECT s3_key FROM results WHERE id = %s", (result_id,))
    row = cursor.fetchone()

    if row:
        s3.delete_object(Bucket=S3_BUCKET, Key=row[0])

    cursor.execute("DELETE FROM results WHERE id = %s", (result_id,))
    conn.commit()
    conn.close()

    return RedirectResponse(url="/dashboard", status_code=302)
# -----------------------------
# EXPORT
# -----------------------------
@app.get("/export")
def export_csv():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT resume_name, match_score, status
        FROM results
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Resume Name", "Match Score", "Status"])

    # Correct dictionary access
    for row in rows:
        writer.writerow([
            row["resume_name"],
            float(row["match_score"]),
            row["status"]
        ])

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=hireSmart_results.csv"
        }
    )
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


#-----------------------------
#candidate-register
#----------------------------

@app.get("/candidate-register", response_class=HTMLResponse)
def candidate_register_page(request: Request):
    return templates.TemplateResponse(
        "candidate_register.html",
        {"request": request}
    )


@app.post("/candidate-register")
def candidate_register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):

    hashed_password = pwd_context.hash(password)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO candidates (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        conn.commit()
    except:
        conn.close()
        return templates.TemplateResponse(
            "candidate_register.html",
            {"request": request, "error": "Email already exists"}
        )

    conn.close()

    return RedirectResponse(url="/candidate-login", status_code=302)


#-------------------------------
#candidate login
#-----------------------------------



@app.get("/candidate-login", response_class=HTMLResponse)
def candidate_login_page(request: Request):
    return templates.TemplateResponse(
        "candidate_login.html",
        {"request": request}
    )
@app.post("/candidate-login")
def candidate_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, password FROM candidates WHERE email = %s",
        (email,)
    )
    user = cursor.fetchone()
    conn.close()

    if not user:
        return templates.TemplateResponse(
            "candidate_login.html",
            {"request": request, "error": "Invalid email or password"}
        )

    user_id = user["id"]
    hashed_password = user["password"]

    if not pwd_context.verify(password, hashed_password):
        return templates.TemplateResponse(
            "candidate_login.html",
            {"request": request, "error": "Invalid email or password"}
        )

    # Store session
    request.session["user"] = email
    request.session["role"] = "candidate"
    request.session["candidate_id"] = user_id

    return RedirectResponse(url="/candidate-dashboard", status_code=302)
#------------------------------------------
#candidat_dashboard
#---------------------------------------
@app.get("/candidate-dashboard", response_class=HTMLResponse)
def candidate_dashboard(request: Request):

    # Only candidate can access
    if request.session.get("role") != "candidate":
        return RedirectResponse(url="/candidate-login", status_code=302)

    candidate_id = request.session.get("candidate_id")

    conn = get_connection()
    cursor = conn.cursor()

    # Get all jobs
    cursor.execute("SELECT id, title, description FROM jobs ORDER BY created_at DESC")
    jobs = cursor.fetchall()

    # Get candidate applications
    cursor.execute("""
        SELECT a.id, j.title, a.match_score, a.status, a.notes
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        WHERE a.candidate_id = %s
        ORDER BY a.applied_at DESC
    """, (candidate_id,))

    applications = cursor.fetchall()
    conn.close()

    return templates.TemplateResponse(
        "candidate_dashboard.html",
        {
            "request": request,
            "jobs": jobs,
            "applications": applications
        }
    )
#---------------------------------------
#candidate apply
#-----------------------------------------


@app.get("/apply/{job_id}", response_class=HTMLResponse)
def apply_page(request: Request, job_id: int):

    if request.session.get("role") != "candidate":
        return RedirectResponse(url="/candidate-login", status_code=302)

    return templates.TemplateResponse(
        "apply.html",
        {
            "request": request,
            "job_id": job_id
        }
    )



@app.get("/apply/{job_id}", response_class=HTMLResponse)
def apply_page(request: Request, job_id: int):

    if request.session.get("role") != "candidate":
        return RedirectResponse(url="/candidate-login", status_code=302)

    return templates.TemplateResponse(
        "apply.html",
        {
            "request": request,
            "job_id": job_id
        }
    )
