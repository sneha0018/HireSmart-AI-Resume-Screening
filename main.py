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
import uuid
import smtplib
from email.mime.text import MIMEText

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

    # Read file
    file_bytes = await file.read()

    # Check file type
    if not file.filename.endswith(".pdf"):
        return templates.TemplateResponse(
            "match.html",
            {
                "request": request,
                "error": "Only PDF files are allowed."
            }
        )

    # Check file size (2MB limit)
    if len(file_bytes) > 2 * 1024 * 1024:
        return templates.TemplateResponse(
            "match.html",
            {
                "request": request,
                "error": "File size must be less than 2MB."
            }
        )

    # Generate unique filename
    unique_name = f"{uuid.uuid4()}.pdf"
    s3_key = f"resumes/{unique_name}"

    # Upload to S3
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
    candidate_id = request.session.get("candidate_id")

    cursor.execute("""
        INSERT INTO applications 
        (candidate_id, job_id, resume_name, s3_key, match_score, status)
        VALUES (%s, %s, %s, %s, %s, %s)
    """,
    (
        candidate_id,
        job_id,
        file.filename,
        s3_key,
        final_score,
        status
    ))

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
        SELECT
            a.id,
            c.name,
            c.email,
            a.resume_name,
            a.match_score,
            a.status,
            a.notes,
            j.title
        FROM applications a
        JOIN candidates c ON a.candidate_id = c.id
        JOIN jobs j ON a.job_id = j.id
    """

    conditions = []
    values = []

    # Search filter
    if search:
        conditions.append("(a.resume_name ILIKE %s OR j.title ILIKE %s)")
        values.extend([f"%{search}%", f"%{search}%"])

    # Job filter
    if selected_job:
        conditions.append("a.job_id = %s")
        values.append(selected_job)

    # Apply conditions
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # Sorting
    if sort == "score":
        query += " ORDER BY r.match_score DESC"
    else:
        query += "ORDER BY a.applied_at DESC"

    # Execute
    cursor.execute(query, values)
    results = cursor.fetchall()
# -------------------------
# Dashboard Stats
# -------------------------

    total_resumes = len(results)
    selected_count = len([r for r in results if r["status"] == "Selected"])
    rejected_count = len([r for r in results if r["status"] == "Rejected"])
    conn.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "results": results,
            "jobs": jobs,
            "selected_job": selected_job,
            "total_resumes": total_resumes,
            "selected_count": selected_count,
            "rejected_count": rejected_count
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

    # Total Applications
    cursor.execute("SELECT COUNT(*) AS total FROM applications")
    total_applications = cursor.fetchone()["total"]

    # Total Jobs
    cursor.execute("SELECT COUNT(*) AS total FROM jobs")
    total_jobs = cursor.fetchone()["total"]

    # Total Candidates
    cursor.execute("SELECT COUNT(*) AS total FROM candidates")
    total_candidates = cursor.fetchone()["total"]

    # Status Breakdown
    cursor.execute("""
        SELECT status, COUNT(*) AS count
        FROM applications
        GROUP BY status
    """)
    status_data = cursor.fetchall()

    # Applications Per Job
    cursor.execute("""
        SELECT j.title, COUNT(a.id) AS count
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        GROUP BY j.title
    """)
    job_data = cursor.fetchall()
    # -------------------------
# Status Counts
# -------------------------

    # -------------------------
# Status Counts
# -------------------------

    cursor.execute("SELECT COUNT(*) FROM applications")
    total_resumes = list(cursor.fetchone().values())[0]

    cursor.execute("SELECT COUNT(*) FROM applications WHERE status = 'Selected'")
    selected_count = list(cursor.fetchone().values())[0]

    cursor.execute("SELECT COUNT(*) FROM applications WHERE status = 'Rejected'")
    rejected_count = list(cursor.fetchone().values())[0]

    conn.close()

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "total_resumes": total_resumes,
            "selected_count": selected_count,
            "rejected_count": rejected_count,
            "job_data": job_data

        }
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

@app.get("/candidate-dashboard")
def candidate_dashboard(request: Request, search: str = None, status: str = None):
    
    candidate_id = request.session.get("candidate_id")
    status_filter = status

    conn = get_connection()
    cursor = conn.cursor()

    # Get all jobs
    if search:
        cursor.execute(
            "SELECT * FROM jobs WHERE LOWER(title) LIKE %s",
            (f"%{search.lower()}%",)
        )
    else:
        cursor.execute("SELECT * FROM jobs")

    jobs = cursor.fetchall()

    # Get only this candidate's applications
    if status_filter and status_filter != "All":
        cursor.execute("""
            SELECT
                a.id AS application_id,
                j.title,
                a.match_score,
                a.status,
                a.notes
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            WHERE a.candidate_id = %s AND a.status = %s
            ORDER BY a.applied_at DESC
        """, (candidate_id, status_filter))
    else:
        cursor.execute("""
            SELECT
                a.id AS application_id,
                j.title,
                a.match_score,
                a.status,
                a.notes
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            WHERE a.candidate_id = %s
            ORDER BY a.applied_at DESC
        """, (candidate_id,))

    applications = cursor.fetchall()


# ===============================
# ✅ ADD RECOMMENDED JOBS HERE
# ===============================

    cursor.execute("""
        SELECT
            j.title,
            a.match_score
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        WHERE a.candidate_id = %s
        ORDER BY a.match_score DESC
        LIMIT 3
    """, (candidate_id,))

    recommended_jobs = cursor.fetchall()

# --------------------------
# Applied Job IDs
# --------------------------
    cursor.execute("""
        SELECT job_id FROM applications
        WHERE candidate_id = %s
    """, (candidate_id,))

    applied_jobs = [row["job_id"] for row in cursor.fetchall()]
# --------------------------
# Dashboard Stats
# --------------------------
    total_applications = len(applications)
    selected_count = len([a for a in applications if a["status"] == "Selected"])
    rejected_count = len([a for a in applications if a["status"] == "Rejected"])
    pending_count = total_applications - selected_count - rejected_count
    # Get job_ids already applied by this candidate
    cursor.execute("""
        SELECT job_id FROM applications
        WHERE candidate_id = %s
    """, (candidate_id,))

    applied_jobs = [row["job_id"] for row in cursor.fetchall()]

    conn.close()

    return templates.TemplateResponse(
        "candidate_dashboard.html",
        {
            "request": request,
            "jobs": jobs,
            "applications": applications,
            "applied_jobs": applied_jobs, 
            "total_applications": total_applications,
            "selected_count": selected_count,
            "recommended_jobs": recommended_jobs, 
            "rejected_count": rejected_count,
            "pending_count": pending_count
        }
    )

#------------------------------------------
#CANDIDATE-PROFILE
#--------------------------------------------



@app.get("/candidate-profile", response_class=HTMLResponse)
def candidate_profile(request: Request):

    if request.session.get("role") != "candidate":
        return RedirectResponse(url="/candidate-login", status_code=302)

    candidate_id = request.session.get("candidate_id")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, email FROM candidates WHERE id = %s",
        (candidate_id,)
    )
    candidate = cursor.fetchone()

    conn.close()

    return templates.TemplateResponse(
        "candidate_profile.html",
        {
            "request": request,
            "candidate": candidate
        }
    )

#---------------------------------
#UPDATE-PROFILE
#------------------------------


@app.post("/update-profile")
def update_profile(
    request: Request,
    name: str = Form(...),
    password: str = Form(None)
):

    if request.session.get("role") != "candidate":
        return RedirectResponse(url="/candidate-login", status_code=302)

    candidate_id = request.session.get("candidate_id")

    conn = get_connection()
    cursor = conn.cursor()

    if password:
        hashed_password = pwd_context.hash(password)
        cursor.execute(
            "UPDATE candidates SET name = %s, password = %s WHERE id = %s",
            (name, hashed_password, candidate_id)
        )
    else:
        cursor.execute(
            "UPDATE candidates SET name = %s WHERE id = %s",
            (name, candidate_id)
        )

    conn.commit()
    conn.close()

    return RedirectResponse(url="/candidate-profile", status_code=302)


#----------------------------------
#withdraw
#-------------------------------------


@app.get("/withdraw/{application_id}")
def withdraw_application(request: Request, application_id: int):

    if request.session.get("role") != "candidate":
        return RedirectResponse(url="/candidate-login", status_code=302)

    candidate_id = request.session.get("candidate_id")

    conn = get_connection()
    cursor = conn.cursor()

    # Ensure candidate owns this application
    cursor.execute("""
        DELETE FROM applications
        WHERE id = %s AND candidate_id = %s
    """, (application_id, candidate_id))

    conn.commit()
    conn.close()

    return RedirectResponse(url="/candidate-dashboard", status_code=302)
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

#-----------------------------
# GMAINL
#------------------------------

def send_status_email(to_email, candidate_name, job_title, status):

    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")

    subject = f"Application Status Update - {job_title}"

    body = f"""
    Hello {candidate_name},

    Your application for the job '{job_title}' has been updated.

    Current Status: {status}

    Thank you for using HireSmart AI.

    Regards,
    HireSmart Team
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print("Email failed:", e)
#---------------------------------------------------------
#UPDAT CANDIDATE APPLICATION
#---------------------------------------------------------


@app.post("/update-candidate/{application_id}")
async def update_candidate(
    request: Request,
    application_id: int,
    status: str = Form(...),
    notes: str = Form(None)
):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=302)

    conn = get_connection()
    cursor = conn.cursor()

    # Get candidate email + name + job title
    cursor.execute("""
        SELECT c.email, c.name, j.title
        FROM applications a
        JOIN candidates c ON a.candidate_id = c.id
        JOIN jobs j ON a.job_id = j.id
        WHERE a.id = %s
    """, (application_id,))

    data = cursor.fetchone()

    if data:
        email = data["email"]
        name = data["name"]
        job_title = data["title"]

    # Update status
    cursor.execute("""
        UPDATE applications
        SET status = %s,
            notes = %s
        WHERE id = %s
    """, (status, notes, application_id))

    conn.commit()
    conn.close()

    print("EMAIL FUNCTION TRIGGERED")
    print("Email:", email)
    print("Status:", status)
    # Send Email
    send_status_email(email, name, job_title, status)

    return RedirectResponse(url="/dashboard", status_code=302)
#-------------------------------
#HOME///////
#-------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request}
    )
