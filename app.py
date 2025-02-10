import streamlit as st
import transformers
import torch
from pylatex import Document, Section, Itemize
import os

# Set up the Streamlit app
st.set_page_config(page_title="CV Tailoring App", page_icon="📄", layout="centered")

# Custom CSS for a dark purple theme with Lucida font
st.markdown(
    """
    <style>
    /* Main container */
    .stApp {
        background-color: #1e1a2f;
        color: #ffffff;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    /* Title */
    h1 {
        color: #bb86fc;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    /* Subheaders */
    h2 {
        color: #bb86fc;
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 20px;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    /* Text area */
    .stTextArea textarea {
        height: 150px;
        border-radius: 8px;
        border: 2px solid #bb86fc;
        padding: 10px;
        font-size: 16px;
        background-color: #2a2342;
        color: #ffffff;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    /* Buttons */
    .stButton button {
        width: 100%;
        background-color: #bb86fc;
        color: #1e1a2f;
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    .stButton button:hover {
        background-color: #9a67ea;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #03dac6;
        color: #1e1a2f;
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    .stDownloadButton button:hover {
        background-color: #018786;
    }
    /* Cards for results */
    .card {
        background-color: #2a2342;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    /* Progress bar for chances */
    .stProgress > div > div > div {
        background-color: #bb86fc;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #bb86fc;
        margin-top: 30px;
        font-size: 0.9rem;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    /* General text */
    body, p, div, span, li {
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize the Llama-3.3-70B-Instruct model
model_id = "meta-llama/Llama-3.3-70B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Function to extract keywords and missing skills
def extract_keywords(job_description):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts keywords and suggests missing skills for a CV."},
            {"role": "user", "content": f"Extract keywords from this job description and suggest missing skills for the CV:\n{job_description}"},
        ]
        response = pipeline(
            messages,
            max_new_tokens=100,
        )
        return response[0]["generated_text"][-1]["content"]
    except Exception as e:
        st.error(f"An error occurred while extracting keywords: {e}")
        return None

# Function to calculate chances of acceptance
def calculate_chances(cv, job_description):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that evaluates CVs against job descriptions."},
            {"role": "user", "content": f"Evaluate the CV against the job description and assign a score for skills match, experience, and education:\nCV: {cv}\nJob Description: {job_description}"},
        ]
        response = pipeline(
            messages,
            max_new_tokens=100,
        )
        evaluation = response[0]["generated_text"][-1]["content"]
        # Example logic to calculate a score (replace with your own logic)
        skills_match = 80  # Example score
        experience = 70    # Example score
        education = 90     # Example score
        overall_score = (skills_match * 0.5) + (experience * 0.3) + (education * 0.2)
        return overall_score
    except Exception as e:
        st.error(f"An error occurred while calculating chances: {e}")
        return 0

# Function to generate a LaTeX CV
def generate_cv(keywords):
    try:
        doc = Document()
        with doc.create(Section('Skills')):
            with doc.create(Itemize()) as itemize:
                itemize.add_item("Python")
                itemize.add_item("Machine Learning")
                itemize.add_item("Data Analysis")
                for skill in keywords.split("\n"):
                    itemize.add_item(skill)
        doc.generate_pdf('tailored_cv', clean_tex=True)
    except Exception as e:
        st.error(f"An error occurred while generating the CV: {e}")

# App Title and Description
st.title("📄 CV Tailoring App")
st.write(
    "Paste a job description, and this app will help you tailor your CV to match the job requirements. "
    "It will also calculate your chances of getting the job."
)

# Input Field for Job Description
st.subheader("🔍 Paste the Job Description")
job_description = st.text_area("", placeholder="Enter the job description here...", label_visibility="collapsed")

# Tailor CV Button
if st.button("✨ Tailor CV"):
    if not job_description:
        st.error("Please paste a job description before tailoring your CV.")
    else:
        with st.spinner("🚀 Tailoring your CV..."):
            # Extract keywords and missing skills
            keywords = extract_keywords(job_description)
            if keywords:
                st.success("✅ CV tailored successfully!")

                # Display extracted keywords and missing skills in a card
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("🔑 Extracted Keywords and Missing Skills")
                    st.write(keywords)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Calculate chances of acceptance
                chances = calculate_chances("Your CV content here", job_description)
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("🎯 Chances of Acceptance")
                    st.write(f"Based on your CV and the job description, your chances are: {chances}%")
                    st.progress(chances / 100)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Generate a LaTeX CV
                generate_cv(keywords)
                st.write("📄 A tailored CV has been generated. Click below to download it.")
                with open("tailored_cv.pdf", "rb") as file:
                    st.download_button(
                        label="⬇️ Download Tailored CV",
                        data=file,
                        file_name="tailored_cv.pdf",
                        mime="application/pdf",
                    )

# Footer
st.markdown("<div class='footer'>Built with ❤️ by Youssef Tati </div>", unsafe_allow_html=True)