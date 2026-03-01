🚀 HireSmart AI – Resume Screening Platform
📌 Project Overview

HireSmart AI is a full-stack AI-powered Resume Screening Platform built using FastAPI and PostgreSQL, deployed on AWS EC2.

The system allows:

Candidates to upload resumes and apply for jobs

Recruiters to evaluate applications using AI-based match scoring

Automatic email notifications when application status changes

Real-time analytics dashboard

🏗 System Architecture

Users (Recruiter / Candidate)
⬇
Browser (HTTP Request)
⬇
AWS EC2 (Ubuntu Server)
⬇
FastAPI Backend (Port 8000 – Uvicorn)
⬇
PostgreSQL Database
⬇
Gmail SMTP (Email Notifications)

Public URL:

http://13.63.138.47

Internal App Port:

127.0.0.1:8000

Managed using:

systemd service → hiresmart.service
🛠 Tech Stack

Backend:

FastAPI

Python 3.12

Uvicorn

Database:

PostgreSQL

Frontend:

HTML

CSS

Deployment:

AWS EC2 (Ubuntu)

systemd

Email:

Gmail SMTP

App Password Authentication

📂 Project Structure
HireSmart-AI-Resume-Screening/
│
├── main.py
├── templates/
│   ├── home.html
│   ├── dashboard.html
│   ├── analytics.html
│   ├── login.html
│   ├── candidate-dashboard.html
│
├── static/
│   └── style.css
│
├── venv/
└── README.md
👥 User Roles
👨‍💼 Recruiter

Login

View all applications

Filter by job

Search resumes

Update candidate status

Add notes

Delete applications

Export CSV

View analytics dashboard

👩‍💻 Candidate

Register

Login

Upload resume

Select job

View application status

Receive email notifications

🔄 Application Workflow
Candidate Flow

Open homepage

Choose “Candidate Login”

Upload resume

Select job

AI score generated

Application stored in database

Recruiter Flow

Login

Open Dashboard

Review applications

Change status (Applied → Shortlisted → Interview → Offered → Hired/Rejected)

Email automatically sent

Analytics updates

📊 Features Implemented

Authentication:

Role-based login

Session management

Resume System:

PDF upload

AI match scoring

Database storage

Dashboard:

Search

Filtering

Status update

Notes

Delete

CSV Export

Analytics:

Total applications

Selected count

Rejected count

Applications per job

Chart visualization

Email:

Gmail SMTP integration

Trigger on status update

Landing Page:

Role selection (Recruiter / Candidate)

Deployment:

AWS EC2

systemd service auto-start

🗄 Database Design

Main Tables:

candidates

id

name

email

password

recruiters

id

email

password

jobs

id

title

description

applications

id

candidate_id

job_id

resume_name

match_score

status

notes

applied_at

Relationships:

applications → linked to candidates & jobs

⚙️ Important Commands (VERY IMPORTANT)
Activate Environment
cd HireSmart-AI-Resume-Screening
source venv/bin/activate
Restart Application
sudo systemctl restart hiresmart
Reload Service (if service file changed)
sudo systemctl daemon-reload
sudo systemctl restart hiresmart
Check Logs

Last 50 lines:

sudo journalctl -u hiresmart -n 50 --no-pager

Live logs:

sudo journalctl -u hiresmart -f
Check Status
sudo systemctl status hiresmart
📧 Email Configuration

SMTP Settings:

Server: smtp.gmail.com
Port: 587
TLS: Enabled
Authentication: Gmail App Password

Email function:

send_status_email()

Triggered when recruiter updates application status.

🔐 Security Notes

Currently implemented:

Basic session-based authentication

App password for Gmail

Recommended improvements:

Use .env file for secrets

Hash passwords securely

Add HTTPS (SSL certificate)

Add CSRF protection

☁️ Deployment Details

Hosted on:
AWS EC2 (Ubuntu)

Process Manager:
systemd

Service file location:

/etc/systemd/system/hiresmart.service

Public IP:

13.63.138.47
🔁 How To Restart Project After 1 Year

Step 1 – Start EC2 instance
Step 2 – SSH:

ssh -i key.pem ubuntu@13.63.138.47

Step 3 – Go to project:

cd HireSmart-AI-Resume-Screening
source venv/bin/activate

Step 4 – Restart app:

sudo systemctl restart hiresmart

Step 5 – Open in browser:

http://13.63.138.47
📈 Project Completion Status

MVP Level: 90%
Production Ready: 65%
Enterprise Level: 30%

🚀 Future Enhancements

Security:

HTTPS (Nginx + SSL)

Environment variable management

Password reset

Features:

Resume keyword extraction

Interview scheduling

Bulk email system

Pagination

Advanced filtering

S3 file storage

Docker containerization

JWT authentication

Role middleware

Admin activity logs

DevOps:

CI/CD pipeline

Docker deployment

Domain integration

💼 Resume Description

Built and deployed an AI-powered Resume Screening Platform using FastAPI, PostgreSQL, and AWS EC2. Implemented role-based authentication, resume-job matching, recruiter analytics dashboard, and automated email notifications. Managed deployment using systemd and Linux server configuration.

👩‍💻 Author

Sneha S
Cloud + AI Enthusiast
AWS | FastAPI | PostgreSQL | Deployment | System Design
