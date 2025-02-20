# STAGI

STAGI is a web app designed to assist with internship applications by tailoring CVs using Large Language Models (LLMs) and Cache Augmented Generation (CAG). It integrates with DeepSeek API for keyword extraction from job descriptiions , extract key skills and refer to a user profile skills database to optimise the cv for that specific application requirments based on the skills database , then generates a tailored cv , and uses Notion API for submissions storage(Job title , company , short description , date , tailored cv etc).

## Features
- Paste a job description to extract keywords.
- Tailor your CV based on the job description (with reference to the skills u have).
- Save submissions to a Notion database.

## Technologies Used
- Python
- Streamlit
- DeepSeek API
- Notion API
- LaTeX

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.

1. Core Workflow Optimization
Preprocessing: Use NLP to clean and standardize job descriptions before keyword extraction.
Keyword Matching: Implement a ranking system to weigh extracted skills against user profiles.
CV Generation: Use a template-based approach with LaTeX for formatting and consistency.
Storage & Tracking: Implement tagging for saved applications to track submission status.
2. Technical Enhancements
Caching Strategy: Use CAG to store processed job descriptions and common skills for efficiency.
Parallel Processing: Speed up API calls using async requests for DeepSeek and Notion.
User Profiles: Store user skills dynamically and allow manual updates for accuracy.
3. UI/UX Improvements
Streamlined Input: Auto-detect job descriptions from LinkedIn or PDFs.
Preview Mode: Show CV modifications before finalizing.
Dashboard: Track past applications with filters (date, company, skills matched).
4. Future Scalability
Multi-Resume Support: Store multiple versions of a user’s CV for different industries.
API Expansion: Consider adding GPT models for better CV phrasing.
Integrations: Automate job scraping from platforms (LinkedIn, Indeed).