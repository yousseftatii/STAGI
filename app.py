

import streamlit as st
import os
import json
from utils.keyword_extraction import extract_keywords
from utils.cv_generation import generate_tailored_cv

# Set up the Streamlit app
st.set_page_config(page_title="CV Tailoring App", page_icon="📄", layout="centered")

# Custom CSS for a dark purple theme with Lucida font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lucida+Sans&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Lucida Sans', sans-serif;
    }
    
    .main {
        background-color: #2A0944;
        color: #FFFFFF;
    }
    
    .stTextArea textarea {
        background-color: #3B185F !important;
        color: #FFFFFF !important;
    }
    
    .stButton button {
        background-color: #A12568 !important;
        color: white !important;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        transition: 0.3s;
    }
    
    .stButton button:hover {
        background-color: #C70A80 !important;
        transform: scale(1.05);
    }
    
    .card {
        background-color: #3B185F !important;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .footer {
        text-align: center;
        color: #A12568;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title and Description
st.title("📄 CV Tailoring App")
st.markdown("""
<div style="border-left: 4px solid #A12568; padding-left: 1rem; margin: 2rem 0;">
    Paste a job description to generate a tailored CV using DeepSeek's AI analysis.
    <br>Focusing on keyword extraction and CV generation testing.
</div>
""", unsafe_allow_html=True)

# API Key Input (with default value)
api_key = st.text_input(
    "🔑 DeepSeek API Key", 
    value=DEEPSEEK_API_KEY,
    type="password",
    help="Get your API key from DeepSeek's platform"
)

# Input Field for Job Description
st.subheader("🔍 Job Description Input")
job_description = st.text_area(
    "Paste job description here:", 
    height=200,
    placeholder="Paste the job description text here..."
)

# Tailor CV Button
if st.button("✨ Generate Tailored CV", use_container_width=True):
    if not api_key:
        st.error("❌ Please enter your DeepSeek API key")
    elif not job_description:
        st.error("❌ Please paste a job description")
    else:
        with st.spinner("🚀 Analyzing job description and generating CV..."):
            try:
                # Step 1: Extract keywords
                keywords = extract_keywords(job_description)
                
                if not keywords:
                    st.error("❌ Failed to extract keywords")
                    st.stop()

                # Display extracted keywords
                with st.expander("🔍 Extracted Keywords", expanded=True):
                    st.json(keywords)

                # Step 2: Generate CV
                cv_path = generate_tailored_cv(job_description, api_key)
                
                if cv_path and os.path.exists(cv_path):
                    # Show success message
                    st.success("✅ CV generated successfully!")
                    
                    # Download button
                    with open(cv_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Tailored CV (PDF)",
                            data=f,
                            file_name="tailored_cv.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    # Show raw LaTeX output
                    with st.expander("📄 Generated LaTeX Code"):
                        tex_path = cv_path.replace(".pdf", ".tex")
                        if os.path.exists(tex_path):
                            with open(tex_path, "r") as f:
                                st.code(f.read(), language="latex")
                        else:
                            st.warning("LaTeX source file not found")
                else:
                    st.error("❌ Failed to generate CV file")

            except Exception as e:
                st.error(f"❌ Error during CV generation: {str(e)}")
                st.stop()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    Built with ❤️ by Youssef Tati | Powered by DeepSeek AI
</div>
""", unsafe_allow_html=True)
