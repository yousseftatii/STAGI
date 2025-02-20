"""
CV Generation Module for STAGI Application
Uses LLM APIs + Keyword Analysis to tailor CVs from LaTeX template
"""

# ---------- IMPORTS ----------
import json
import re
import requests
import subprocess
import os
import tempfile
from typing import Dict, List

# ---------- CONSTANTS ----------
USER_PROFILE_PATH = "utils/user_profile.json"
LATEX_TEMPLATE_PATH = "cv_template.tex"
OUTPUT_DIR = "generated_cvs"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ---------- CORE FUNCTIONS ----------

def load_profile_data() -> Dict:
    """
    Load user profile from JSON file
    Returns: Dict with all user profile data
    """
    try:
        with open(USER_PROFILE_PATH, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading profile data: {e}")
        return {}

def extract_job_keywords(job_description: str, api_key: str) -> Dict:
    """
    Get analyzed keywords from job description using DeepSeek API
    Args:
        job_description: Raw text from job posting
        api_key: DeepSeek API key
    Returns: Dict of extracted keywords
    """
    prompt = f"""Analyze this job description and extract:
    1. Technical skills (programming languages, frameworks)
    2. Tools and technologies
    3. Domain knowledge areas
    4. Experience level (junior/mid/senior)
    5. Potentially missing skills from my profile
    
    Return JSON format with keys: technical_skills, tools, domains, experience_level, missing_skills
    
    Job description: {job_description[:3000]}"""  # Truncate to avoid token limits

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content']
        return json.loads(result.strip('` \n'))
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return {
            "technical_skills": [],
            "tools": [],
            "domains": [],
            "experience_level": "",
            "missing_skills": []
        }

def analyze_keyword_match(user_profile: Dict, job_keywords: Dict) -> Dict:
    """
    Identify matches between user profile and job requirements
    """
    # Extract user competencies from profile
    user_skills = {
        'languages': user_profile['competencies']['HIGHLIGHTED_LANGUAGES'].split(', '),
        'data_tools': user_profile['competencies']['RELEVANT_DATA_TOOLS'].split(', '),
        'ml_skills': user_profile['competencies']['ML_AI_SKILLS'].split(', '),
        'soft_skills': user_profile['competencies']['RELEVANT_SOFT_SKILLS'].split(', ')
    }

    # Flatten job keywords
    job_skills = set(
        job_keywords['technical_skills'] +
        job_keywords['tools'] +
        job_keywords['domains']
    )

    # Find matches and gaps
    matched_skills = []
    for category in user_skills.values():
        matched_skills.extend([skill for skill in category if skill in job_skills])
    
    # Get all user skills for missing skills check
    all_user_skills = [skill for sublist in user_skills.values() for skill in sublist]
    missing_skills = [skill for skill in job_keywords['missing_skills'] if skill not in all_user_skills]

    # Simple relevance scoring
    skill_gap_analysis = {skill: 1.0 for skill in matched_skills}
    for skill in missing_skills:
        skill_gap_analysis[skill] = 0.0

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_gap_analysis": skill_gap_analysis,
        "priority_skills": sorted(matched_skills, key=lambda x: -skill_gap_analysis.get(x, 0))
    }

def generate_tailored_summary(user_profile: Dict, job_keywords: Dict, analysis: Dict, api_key: str) -> str:
    """
    Generate dynamic professional summary using DeepSeek API
    """
    base_summary = user_profile['summary']['TAILORED_SUMMARY']
    prompt = f"""Create a 2-sentence professional summary for a CV that:
    - Highlights these key skills: {', '.join(analysis['priority_skills'][:5])}
    - Mentions experience with: {', '.join(job_keywords['domains'][:3])}
    - Matches experience level: {job_keywords['experience_level']}
    - Maintains truthfulness with this base summary: "{base_summary}"
    
    Use natural language and avoid markdown formatting."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        summary = response.json()['choices'][0]['message']['content']
        return summary.strip('"').replace('\n', ' ')
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return base_summary  # Fallback to original summary

def highlight_relevant_skills(user_skills: List[str], priority_skills: List[str], user_profile: Dict) -> str:
    """
    Reorder and format skills list based on job relevance
    """
    # Create priority-ordered set without duplicates
    ordered_skills = [s for s in priority_skills if s in user_skills]
    
    # Add remaining skills in original order
    for skill in user_skills:
        if skill not in ordered_skills:
            ordered_skills.append(skill)
    
    # Group skills into categories based on original profile structure
    categorized = {
        "Programming Languages": [],
        "Data Tools": [],
        "ML/AI": [],
        "Soft Skills": []
    }
    
    for skill in ordered_skills:
        if skill in user_profile['competencies']['HIGHLIGHTED_LANGUAGES']:
            categorized["Programming Languages"].append(skill)
        elif skill in user_profile['competencies']['RELEVANT_DATA_TOOLS']:
            categorized["Data Tools"].append(skill)
        elif skill in user_profile['competencies']['ML_AI_SKILLS']:
            categorized["ML/AI"].append(skill)
        elif skill in user_profile['competencies']['RELEVANT_SOFT_SKILLS']:
            categorized["Soft Skills"].append(skill)
    
    # Build LaTeX-formatted string
    latex_skills = []
    for category, skills in categorized.items():
        if skills:
            latex_skills.append(f"\\textbf{{{category}}}: {', '.join(skills)}")
    
    return "\n\n\\vspace{0.1 cm}\n\n".join(latex_skills)

def generate_latex_content(template: str, data: Dict) -> str:
    """
    Populate LaTeX template with tailored content
    """
    replacements = {
        "{{TAILORED_SUMMARY}}": _sanitize_latex(data["summary"]),
        "{{HIGHLIGHTED_LANGUAGES}}": _sanitize_latex(data["competencies"]["languages"]),
        "{{RELEVANT_DATA_TOOLS}}": _sanitize_latex(data["competencies"]["data_tools"]),
        "{{ML_AI_SKILLS}}": _sanitize_latex(data["competencies"]["ml_skills"]),
        "{{RELEVANT_SOFT_SKILLS}}": _sanitize_latex(data["competencies"]["soft_skills"]),
        "{{EDUCATION_ENTRIES}}": _format_education_entries(data["education"]),
        "{{EXPERIENCE_ENTRIES}}": _format_experience_entries(data["experience"]),
        "{{PROJECT_ENTRIES}}": _format_project_entries(data["projects"]),
        "{{CERTIFICATION_ENTRIES}}": _format_certifications(data["certifications"]),
        "{{EXTRACURRICULAR_ENTRIES}}": _format_extracurricular(data["extracurricular"])
    }
    
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    
    return template

def compile_latex_to_pdf(tex_content: str, output_name: str) -> str:
    """
    Compile generated LaTeX to PDF
    Returns: Path to generated PDF if successful, None otherwise
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, f"{output_name}.tex")
            with open(tex_path, "w") as f:
                f.write(tex_content)
            
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                final_path = os.path.join(OUTPUT_DIR, f"{output_name}.pdf")
                os.rename(os.path.join(tmpdir, f"{output_name}.pdf"), final_path)
                return final_path
            print(f"LaTeX compilation failed with errors:\n{result.stderr}")
            return None
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None

# ---------- PIPELINE ORCHESTRATION ----------

def generate_tailored_cv(job_description: str, api_key: str) -> str:
    """
    Main CV generation pipeline
    Returns: Path to generated PDF or None
    """
    # 1. Load profile data
    profile = load_profile_data()
    if not profile:
        print("Failed to load profile data")
        return None
    
    # 2. Extract keywords
    job_keywords = extract_job_keywords(job_description, api_key)
    if not job_keywords:
        print("Failed to extract keywords")
        return None
    
    # 3. Analyze matches
    analysis = analyze_keyword_match(profile, job_keywords)
    
    # 4. Generate summary
    summary = generate_tailored_summary(profile, job_keywords, analysis, api_key)
    
    try:
        with open(LATEX_TEMPLATE_PATH, "r") as f:
            template = f.read()
    except FileNotFoundError:
        print(f"LaTeX template not found at {LATEX_TEMPLATE_PATH}")
        return None
    
    # 5. Prepare content
    data = {
        "summary": summary,
        "competencies": {
            "languages": highlight_relevant_skills(
                profile["competencies"]["HIGHLIGHTED_LANGUAGES"].split(", "),
                analysis["priority_skills"],
                profile
            ),
            "data_tools": highlight_relevant_skills(
                profile["competencies"]["RELEVANT_DATA_TOOLS"].split(", "),
                job_keywords["tools"],
                profile
            ),
            "ml_skills": highlight_relevant_skills(
                profile["competencies"]["ML_AI_SKILLS"].split(", "),
                job_keywords["technical_skills"],
                profile
            ),
            "soft_skills": ", ".join(analysis["priority_skills"])
        },
        "education": profile["education"]["EDUCATION_ENTRIES"],
        "experience": profile["experience"]["EXPERIENCE_ENTRIES"],
        "projects": profile["projects"]["PROJECT_ENTRIES"],
        "certifications": profile["certifications"]["CERTIFICATION_ENTRIES"],
        "extracurricular": profile["extracurricular"]["EXTRACURRICULAR_ENTRIES"]
    }
    
    # 6. Fill template
    latex_content = generate_latex_content(template, data)
    
    # 7. Compile PDF
    output_name = re.sub(r"\W+", "", job_keywords.get("experience_level", "cv"))[:50]
    pdf_path = compile_latex_to_pdf(latex_content, output_name)
    
    return pdf_path

# ---------- HELPER FUNCTIONS ----------

def _format_education_entries(education: List[Dict]) -> str:
    """
    Format education entries into LaTeX code
    """
    latex_education = []
    for edu in education:
        # Format highlights separately
        highlights = "\n".join(
            fr"\item {_sanitize_latex(h)}" 
            for h in edu['highlights']
        )
        
        entry = fr"""
\begin{{twocolentry}}{{
    \textit{{{_sanitize_latex(edu['duration'])}}} \\
    \textbf{{{_sanitize_latex(edu['institution'])}}} \\
    \textit{{{_sanitize_latex(edu['degree'])}}}
\end{{twocolentry}}

\vspace{{0.10 cm}}
\begin{{onecolentry}}
    \begin{{highlights}}
        {highlights}
    \end{{highlights}}
\end{{onecolentry}}
\vspace{{0.2 cm}}"""
        latex_education.append(entry)
    return "\n".join(latex_education)

def _format_project_entries(projects: List[Dict]) -> str:
    """
    Format project entries into LaTeX code
    """
    latex_projects = []
    for proj in projects:
        # Format highlights separately
        highlights = "\n".join(
            fr"\item {_sanitize_latex(h)}" 
            for h in proj['highlights']
        )
        
        entry = fr"""
\begin{{twocolentry}}{{
    \textit{{{_sanitize_latex(proj['date'])}}} \\
    \textbf{{{_sanitize_latex(proj['title'])}}}
\end{{twocolentry}}

\vspace{{0.10 cm}}
\begin{{onecolentry}}
    \begin{{highlights}}
        {highlights}
    \end{{highlights}}
\end{{onecolentry}}
\vspace{{0.2 cm}}"""
        latex_projects.append(entry)
    return "\n".join(latex_projects)

def _format_certifications(certifications: List[Dict]) -> str:
    """
    Format certification entries into LaTeX code
    """
    latex_certs = []
    for cert in certifications:
        entry = fr"""
\begin{{twocolentry}}{{
    \textit{{{_sanitize_latex(cert['date'])}}} \\
    \textbf{{{_sanitize_latex(cert['title'])}}}
\end{{twocolentry}}
\vspace{{0.1 cm}}"""
        latex_certs.append(entry)
    return "\n".join(latex_certs)

def _format_extracurricular(activities: List[Dict]) -> str:
    """
    Format extracurricular entries into LaTeX code
    """
    latex_activities = []
    for activity in activities:
        entry = fr"""
\begin{{twocolentry}}{{
    \textit{{{_sanitize_latex(activity['duration'])}}} \\
    \textbf{{{_sanitize_latex(activity['role'])}}} \\
    \textit{{{_sanitize_latex(activity['organization'])}}}
\end{{twocolentry}}

\vspace{{0.10 cm}}
\begin{{onecolentry}}
    {_sanitize_latex(activity['description'])}
\end{{onecolentry}}
\vspace{{0.2 cm}}"""
        latex_activities.append(entry)
    return "\n".join(latex_activities)

def _format_experience_entries(experiences: List[Dict]) -> str:
    """
    Format experience entries into LaTeX code
    """
    latex_experience = []
    for exp in experiences:
        # Format highlights separately
        highlights = "\n".join(
            fr"\item {_sanitize_latex(h)}" 
            for h in exp['highlights']
        )
        
        entry = fr"""
\begin{{twocolentry}}{{
    \textit{{{_sanitize_latex(exp['location'])}}} \\ 
    \textit{{{_sanitize_latex(exp['duration'])}}} }}
    \textbf{{{_sanitize_latex(exp['role'])}}} \\
    \textit{{{_sanitize_latex(exp['organization'])}}}
\end{{twocolentry}}

\vspace{{0.05 cm}}
\begin{{onecolentry}}
    \begin{{highlights}}
        {highlights}
    \end{{highlights}}
    \textbf{{Technologies \& Tools:}} {_sanitize_latex(exp.get('technologies', ''))}
\end{{onecolentry}}
\vspace{{0.2cm}}"""
        latex_experience.append(entry)
    return "\n".join(latex_experience)

def _sanitize_latex(text: str) -> str:
    """
    Sanitize text for LaTeX formatting
    """
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde",
        "^": r"\textasciicircum",
        "\\": r"\textbackslash"
    }
    return "".join(replacements.get(c, c) for c in text)

def _call_llm_api(prompt: str, api_key: str) -> str:
    """
    Call LLM API with the given prompt and API key
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return ""