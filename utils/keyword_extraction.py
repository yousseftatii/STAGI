"""
Keyword Extraction Module for Job Descriptions
Simple but efficient implementation for initial functionality
"""

from typing import List, Tuple, Dict
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests
from collections import defaultdict

# Initialize NLP models
try:
    nlp = spacy.load("en_core_web_sm")  # Lightweight English model
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please install it using 'python -m spacy download en_core_web_sm'.")

tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

# Domain-specific keyword banks (pre-populated with common terms)
TECH_SKILLS_BANK = [
    "python", "machine learning", "data analysis", "sql", "pandas", "numpy", 
    "scikit-learn", "tensorflow", "pytorch", "deep learning", "data visualization", 
    "statistics", "natural language processing", "computer vision", "big data"
]

DS_DOMAINS_BANK = [
    "machine learning", "deep learning", "data engineering", "data science", 
    "business intelligence", "artificial intelligence", "computer vision", 
    "natural language processing", "data analytics"
]

ML_TOOLS_BANK = [
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm", 
    "spark", "hadoop", "tableau", "powerbi", "aws", "azure", "google cloud"
]

def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw job description text.
    Args:
        text: Raw job description from any source.
    Returns:
        Cleaned text ready for processing.
    """
    # Remove HTML/XML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove special characters and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()

def extract_technical_skills(text: str) -> List[Tuple[str, float]]:
    """
    Extract technical skills with confidence scores using rule-based matching.
    Args:
        text: Preprocessed job description.
    Returns:
        List of (skill, confidence_score) tuples.
    """
    skills_found = []
    doc = nlp(text)
    
    # Rule 1: Direct matches from TECH_SKILLS_BANK
    for skill in TECH_SKILLS_BANK:
        if skill in text:
            skills_found.append((skill, 1.0))  # Max confidence for exact matches
    
    # Rule 2: Noun phrases containing skill-related terms
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        for skill in TECH_SKILLS_BANK:
            if skill in chunk_text and skill not in [s[0] for s in skills_found]:
                skills_found.append((skill, 0.8))  # Slightly lower confidence
    
    return skills_found

def extract_tools_technologies(text: str) -> Dict[str, List[str]]:
    """
    Identify specific tools/technologies mentioned with context.
    Args:
        text: Preprocessed job description.
    Returns:
        Categories with tools:
        {
            'programming_languages': [],
            'ml_frameworks': [],
            'cloud_tech': [],
            'databases': [],
            'devops_tools': []
        }
    """
    tools = {
        "programming_languages": [],
        "ml_frameworks": [],
        "cloud_tech": [],
        "databases": [],
        "devops_tools": []
    }
    
    # Programming languages
    for lang in ["python", "java", "r", "scala", "c++", "javascript"]:
        if lang in text:
            tools["programming_languages"].append(lang)
    
    # ML frameworks
    for framework in ["tensorflow", "pytorch", "keras", "scikit-learn", "xgboost"]:
        if framework in text:
            tools["ml_frameworks"].append(framework)
    
    # Cloud technologies
    for cloud in ["aws", "azure", "google cloud", "gcp"]:
        if cloud in text:
            tools["cloud_tech"].append(cloud)
    
    # Databases
    for db in ["sql", "mysql", "postgresql", "mongodb", "cassandra"]:
        if db in text:
            tools["databases"].append(db)
    
    # DevOps tools
    for tool in ["docker", "kubernetes", "jenkins", "terraform", "ansible"]:
        if tool in text:
            tools["devops_tools"].append(tool)
    
    return tools

def extract_domains(text: str) -> List[Tuple[str, float]]:
    """
    Detect domain-specific focus areas from job description.
    Args:
        text: Preprocessed job description.
    Returns:
        List of (domain, relevance_score) tuples.
    """
    domains_found = []
    tfidf_scores = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_scores.toarray()[0]
    
    # Match domains from DS_DOMAINS_BANK
    for domain in DS_DOMAINS_BANK:
        if domain in text:
            # Use TF-IDF score as relevance score
            if domain in feature_names:
                idx = list(feature_names).index(domain)
                score = tfidf_scores[idx]
            else:
                score = 0.5  # Default score for domain matches
            domains_found.append((domain, score))
    
    return domains_found

#####################################################################################################################################

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

# Synonym mapping for skills
SKILL_SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "ai": "artificial intelligence",
    "bi": "business intelligence",
    "db": "database",
}

# Experience level keywords
EXPERIENCE_KEYWORDS = {
    "entry": ["entry-level", "junior", "0-2 years", "fresher", "recent graduate"],
    "mid": ["mid-level", "3-5 years", "experienced", "intermediate"],
    "senior": ["senior", "5+ years", "expert", "leadership"],
    "lead": ["lead", "principal", "manager", "director", "head of"]
}

# Qualification keywords
QUALIFICATION_KEYWORDS = {
    "degrees": ["bachelor", "master", "phd", "bs", "ms", "mtech", "btech"],
    "certifications": ["certified", "certification", "aws certified", "google cloud certified"],
    "courses": ["coursework", "online course", "mooc", "coursera", "udemy"]
}

def calculate_keyword_relevance(text: str, keywords: List[str]) -> Dict[str, float]:
    """
    Calculate contextual relevance scores for keywords using TF-IDF.
    Args:
        text: Full cleaned job description.
        keywords: List of candidate keywords.
    Returns:
        Dictionary of {keyword: relevance_score}.
    """
    # Fit TF-IDF on the job description
    tfidf_scores = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_scores.toarray()[0]
    
    # Map keywords to their TF-IDF scores
    relevance_scores = {}
    for keyword in keywords:
        if keyword in feature_names:
            idx = list(feature_names).index(keyword)
            relevance_scores[keyword] = tfidf_scores[idx]
        else:
            relevance_scores[keyword] = 0.0  # Default score if keyword not found
    
    return relevance_scores

def detect_experience_level(text: str) -> str:
    """
    Identify required experience level from job description using keyword patterns.
    Args:
        text: Preprocessed job description.
    Returns:
        Experience level: 'entry', 'mid', 'senior', 'lead'.
    """
    # Count occurrences of experience-related keywords
    level_counts = defaultdict(int)
    for level, keywords in EXPERIENCE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                level_counts[level] += 1
    
    # Determine the most frequent experience level
    if level_counts:
        return max(level_counts, key=level_counts.get)
    return "mid"  # Default to mid-level if no keywords are found

def identify_required_qualifications(text: str) -> Dict[str, List[str]]:
    """
    Extract educational requirements and certifications using keyword matching.
    Args:
        text: Preprocessed job description.
    Returns:
        {
            'degrees': [],
            'certifications': [],
            'courses': []
        }
    """
    qualifications = {
        "degrees": [],
        "certifications": [],
        "courses": []
    }
    
    # Extract degrees
    for degree in QUALIFICATION_KEYWORDS["degrees"]:
        if degree in text:
            qualifications["degrees"].append(degree)
    
    # Extract certifications
    for cert in QUALIFICATION_KEYWORDS["certifications"]:
        if cert in text:
            qualifications["certifications"].append(cert)
    
    # Extract courses
    for course in QUALIFICATION_KEYWORDS["courses"]:
        if course in text:
            qualifications["courses"].append(course)
    
    return qualifications

def resolve_skill_synonyms(skills: List[str]) -> List[str]:
    """
    Normalize skills/tools to canonical forms using a synonym mapping.
    Args:
        skills: Raw extracted skills list.
    Returns:
        Normalized skills list with synonyms resolved.
    """
    normalized_skills = []
    for skill in skills:
        # Convert to lowercase and check for synonyms
        skill_lower = skill.lower()
        if skill_lower in SKILL_SYNONYMS:
            normalized_skills.append(SKILL_SYNONYMS[skill_lower])
        else:
            normalized_skills.append(skill)
    return normalized_skills

def rank_keywords(keywords: Dict[str, float], 
                 domain: str, 
                 experience_level: str) -> List[Tuple[str, float]]:
    """
    Prioritize keywords based on multiple factors.
    Args:
        keywords: Keywords with relevance scores.
        domain: Primary domain (e.g., 'Machine Learning').
        experience_level: Detected experience level.
    Returns:
        Sorted list of (keyword, priority_score) tuples.
    """
    # Domain relevance multiplier
    domain_multiplier = 1.5 if domain in keywords else 1.0
    
    # Experience level weights
    experience_weights = {
        "entry": 0.8,
        "mid": 1.0,
        "senior": 1.2,
        "lead": 1.5
    }
    experience_weight = experience_weights.get(experience_level, 1.0)
    
    # Calculate priority scores
    priority_scores = []
    for keyword, relevance in keywords.items():
        # Adjust score based on domain and experience level
        priority_score = relevance * domain_multiplier * experience_weight
        priority_scores.append((keyword, priority_score))
    
    # Sort by priority score (descending)
    return sorted(priority_scores, key=lambda x: x[1], reverse=True)


#####################################################################################################################################

from typing import Dict, List
import requests

# Placeholder for DeepSeek API endpoint and key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = "sk-5229cbdfcb984c52b18447b202c464cf"

def identify_missing_skills(extracted_skills: List[str], 
                          user_profile: Dict) -> Dict[str, List[str]]:
    """
    Compare extracted skills with user profile to identify gaps.
    Args:
        extracted_skills: Skills from job description.
        user_profile: User's skills database.
    Returns:
        {
            'missing_hard_skills': [],
            'missing_soft_skills': [],
            'partial_matches': []
        }
    """
    user_hard_skills = set(user_profile.get("hard_skills", []))
    user_soft_skills = set(user_profile.get("soft_skills", []))
    extracted_skills_set = set(extracted_skills)
    
    # Identify missing hard skills
    missing_hard_skills = list(extracted_skills_set - user_hard_skills)
    
    # Identify missing soft skills (if any soft skills are extracted)
    missing_soft_skills = list(extracted_skills_set - user_soft_skills)
    
    # Identify partial matches (skills that are similar but not exact)
    partial_matches = []
    for skill in extracted_skills_set:
        if skill not in user_hard_skills and skill not in user_soft_skills:
            # Check for partial matches (e.g., "machine learning" vs "ml")
            normalized_skill = resolve_skill_synonyms([skill])[0]
            if normalized_skill in user_hard_skills or normalized_skill in user_soft_skills:
                partial_matches.append(skill)
    
    return {
        "missing_hard_skills": missing_hard_skills,
        "missing_soft_skills": missing_soft_skills,
        "partial_matches": partial_matches
    }

def format_keywords_output(processed_data: Dict) -> str:
    """
    Format extracted keywords for display in Streamlit app.
    Args:
        processed_data: Combined extraction results.
    Returns:
        User-friendly formatted string with categorization.
    """
    output = []
    
    # Technical Skills
    if processed_data.get("technical_skills"):
        output.append("### 🔑 Technical Skills")
        output.append("\n".join([f"- {skill} (Score: {score:.2f})" for skill, score in processed_data["technical_skills"]]))
    
    # Tools and Technologies
    if processed_data.get("tools_technologies"):
        output.append("\n### 🛠️ Tools and Technologies")
        for category, tools in processed_data["tools_technologies"].items():
            output.append(f"**{category.title()}:**")
            output.append("\n".join([f"- {tool}" for tool in tools]))
    
    # Domains
    if processed_data.get("domains"):
        output.append("\n### 🌐 Domains")
        output.append("\n".join([f"- {domain} (Relevance: {score:.2f})" for domain, score in processed_data["domains"]]))
    
    # Missing Skills
    if processed_data.get("missing_skills"):
        output.append("\n### ❌ Missing Skills")
        for category, skills in processed_data["missing_skills"].items():
            if skills:
                output.append(f"**{category.title()}:**")
                output.append("\n".join([f"- {skill}" for skill in skills]))
    
    # Priority Keywords
    if processed_data.get("priority_keywords"):
        output.append("\n### 🎯 Priority Keywords")
        output.append("\n".join([f"- {keyword} (Priority: {score:.2f})" for keyword, score in processed_data["priority_keywords"]]))
    
    return "\n".join(output)

def call_deepseek_api(text: str) -> Dict:
    """
    Integration with DeepSeek API for additional validation.
    Args:
        text: Preprocessed job description.
    Returns:
        API response with augmented keyword analysis.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "analysis_type": "keyword_extraction"
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {e}")
        return {}
###############################################################################################################################
def extract_keywords(job_description: str) -> Dict:
    """
    Main function to process job description and return structured keywords.
    Args:
        job_description: Raw input text from user.
    Returns:
        Structured analysis:
        {
            'technical_skills': [],
            'tools_technologies': {},
            'domains': [],
            'experience_level': '',
            'qualifications': {},
            'missing_skills': {},
            'priority_keywords': []
        }
    """
    # Step 1: Preprocess text
    cleaned_text = preprocess_text(job_description)
    
    # Step 2: Extract entities
    technical_skills = extract_technical_skills(cleaned_text)
    tools_technologies = extract_tools_technologies(cleaned_text)
    domains = extract_domains(cleaned_text)
    experience_level = detect_experience_level(cleaned_text)
    qualifications = identify_required_qualifications(cleaned_text)
    
    # Step 3: Calculate relevance
    keywords = [skill for skill, _ in technical_skills]
    relevance_scores = calculate_keyword_relevance(cleaned_text, keywords)
    
    # Step 4: Identify missing skills (requires user profile)
    user_profile = {
        "hard_skills": ["python", "sql", "data analysis"],
        "soft_skills": ["communication", "teamwork"]
    }
    missing_skills = identify_missing_skills(keywords, user_profile)
    
    # Step 5: Rank keywords
    primary_domain = domains[0][0] if domains else "machine learning"
    priority_keywords = rank_keywords(relevance_scores, primary_domain, experience_level)
    
    # Step 6: Call DeepSeek API for additional analysis
    deepseek_analysis = call_deepseek_api(cleaned_text)
    
    # Combine all results
    structured_data = {
        "technical_skills": technical_skills,
        "tools_technologies": tools_technologies,
        "domains": domains,
        "experience_level": experience_level,
        "qualifications": qualifications,
        "missing_skills": missing_skills,
        "priority_keywords": priority_keywords,
        "deepseek_analysis": deepseek_analysis
    }
    
    return structured_data