#!/usr/bin/env python3
"""
Resume Fit Analyzer - Streamlit + LangGraph App
Run with: uv run streamlit run app.py
"""

import os
import tempfile
from typing import TypedDict, Annotated
import operator

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# PYDANTIC MODELS
# ============================================================

class PdfExtractor(BaseModel):
    """Structured extraction from resume PDF."""
    name: str = Field(description="Full name of the applicant")
    age: str = Field(description="Age of the applicant (if mentioned)")
    contact: str = Field(description="Phone number or contact details")
    email: str = Field(description="Email address of the applicant")
    summary: str = Field(description="Professional summary / overall profile overview")
    skills: list[str] = Field(description="List of technical and soft skills")
    work_experience: list[str] = Field(description="List of work experiences with company, role, and key achievements")
    certificates: list[str] = Field(description="List of relevant certificates, courses, or licenses")


class Recommendation(BaseModel):
    """Final rating and feedback."""
    rate: int = Field(description="Rating from 1 to 10 indicating how qualified the applicant is for the role")
    recom: str = Field(description="Detailed assessment, strengths, gaps, and actionable feedback")


class State(TypedDict):
    """LangGraph state."""
    job_role: str
    job_description: str
    resume_path: str
    extracted: str
    output: Annotated[list, operator.add]


# ============================================================
# GRAPH NODES
# ============================================================

def extract_pdf(file_path: str) -> str:
    """Extract clean text from PDF resume."""
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()
    text_content = [page.page_content for page in pages]
    return "\n\n".join(text_content)


def extract_data(state: State) -> dict:
    """Node 1: Extract text from the uploaded resume PDF."""
    extracted_text = extract_pdf(state["resume_path"])
    return {"extracted": extracted_text}


def parse_pdf(state: State) -> dict:
    """Node 2: Parse extracted text into structured Pydantic model."""
    message = structured_ai.invoke(
        [
            SystemMessage(
                "You are a precise resume parser. Extract name, age, contact info, email, "
                "professional summary, skills, work experience, and certificates from the provided resume text."
            ),
            HumanMessage(f"Resume text:\n{state['extracted']}")
        ]
    )
    return {"output": [message]}


def show_result(state: State) -> dict:
    """Node 3: Generate rating (1-10) + detailed feedback using job role + description."""
    job_role = state["job_role"]
    job_desc = state.get("job_description", "")
    applicant_data = state["output"][0] if state.get("output") else "No structured data available"

    system_prompt = """You are an expert HR Resume Analyst and Talent Acquisition Specialist with 10+ years of experience 
in tech, HR, and administrative recruitment.

Your job is to:
1. Carefully compare the candidate's background against the job role and job description.
2. Give an honest integer rating from 1 to 10 (10 = perfect match).
3. Provide clear, constructive, and actionable feedback covering:
   - Key strengths and why they match
   - Important gaps or missing requirements
   - Specific suggestions to improve the resume for this role
   - Overall recommendation (strong fit / moderate fit / needs development)

Be professional, specific, and balanced. Do not be overly harsh or overly generous."""

    human_prompt = f"""Job Role: {job_role}

Job Description:
{job_desc}

Candidate's Structured Resume Data:
{applicant_data}

Please evaluate the candidate and return the rating + detailed feedback."""

    message = structured_ai2.invoke(
        [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt)
        ]
    )
    return {"output": [message]}


# ============================================================
# BUILD LANGGRAPH (runs once at import)
# ============================================================

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # Will be handled gracefully in Streamlit UI
    google_model = None
    structured_ai = None
    structured_ai2 = None
    agent = None
else:
    google_model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",           # ← Change to gemini-2.0-flash or latest if available
        api_key=api_key,
        temperature=1.0
    )
    structured_ai = google_model.with_structured_output(PdfExtractor)
    structured_ai2 = google_model.with_structured_output(Recommendation)

    graph = StateGraph(State)
    graph.add_node("extract_text", extract_data)
    graph.add_node("parse_structured", parse_pdf)
    graph.add_node("analyze_fit", show_result)

    graph.add_edge(START, "extract_text")
    graph.add_edge("extract_text", "parse_structured")
    graph.add_edge("parse_structured", "analyze_fit")
    graph.add_edge("analyze_fit", END)

    agent = graph.compile()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Resume Fit Analyzer",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🎯 AI Resume Fit Analyzer")
st.markdown(
    "Upload your resume (PDF), enter the job role + description, and get an instant "
    "**rating (1-10)** + professional feedback powered by Gemini + LangGraph."
)

st.divider()

# Input form
with st.form(key="analyzer_form", clear_on_submit=False):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        job_role = st.text_input(
            "Job Role / Position Title",
            placeholder="e.g. Senior Full Stack Engineer",
            help="The exact title of the position you're applying for"
        )
    
    with col2:
        st.write("")  # spacing

    job_description = st.text_area(
        "Job Description",
        placeholder="Paste the full job description here (requirements, responsibilities, qualifications...)",
        height=180,
        help="The more detailed the job description, the better the analysis"
    )

    uploaded_file = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=["pdf"],
        help="Your resume is processed locally and never stored permanently.",
        label_visibility="visible"
    )

    submitted = st.form_submit_button(
        "🚀 Analyze Resume Fit",
        use_container_width=True,
        type="primary"
    )

# Processing
if submitted:
    # Validation
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found. Please create a `.env` file with your Google API key.")
        st.stop()
    
    if not job_role.strip():
        st.error("Please enter a Job Role.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste the Job Description.")
        st.stop()
    if uploaded_file is None:
        st.error("Please upload a PDF resume.")
        st.stop()

    # Save uploaded PDF to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        resume_path = tmp_file.name

    try:
        with st.status("🔍 Analyzing your resume with AI...", expanded=True) as status:
            st.write("Step 1/3: Extracting text from PDF...")
            st.write("Step 2/3: Structuring resume data...")
            st.write("Step 3/3: Generating rating & feedback...")

            initial_state = {
                "job_role": job_role.strip(),
                "job_description": job_description.strip(),
                "resume_path": resume_path,
            }

            # Run the LangGraph
            result = agent.invoke(initial_state)

            # Extract results (output list accumulates both parsed data + recommendation)
            parsed_resume: PdfExtractor = result["output"][0]
            recommendation: Recommendation = result["output"][1]

            status.update(label="✅ Analysis complete!", state="complete", expanded=False)

        # ===================== RESULTS =====================
        st.success("Analysis finished successfully!")

        # Rating display
        st.subheader("📊 Match Rating")
        rating = recommendation.rate

        # Color-coded metric
        if rating >= 8:
            delta_text = "Excellent Fit"
            delta_color = "normal"
        elif rating >= 6:
            delta_text = "Good Fit"
            delta_color = "normal"
        elif rating >= 4:
            delta_text = "Moderate Fit"
            delta_color = "off"
        else:
            delta_text = "Needs Improvement"
            delta_color = "off"

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric(
                label="Qualification Score",
                value=f"{rating}/10",
                delta=delta_text,
                delta_color=delta_color
            )
        with col_r2:
            st.progress(rating / 10.0, text=f"Fit Score: {rating} / 10")

        # Feedback
        st.subheader("💡 AI Feedback & Recommendations")
        st.markdown(recommendation.recom)

        # Parsed resume details (collapsible)
        with st.expander("📄 View Parsed Resume Data (for transparency)", expanded=False):
            st.markdown(f"**Name:** {parsed_resume.name}")
            st.markdown(f"**Age:** {parsed_resume.age} &nbsp;&nbsp;|&nbsp;&nbsp; **Email:** {parsed_resume.email} &nbsp;&nbsp;|&nbsp;&nbsp; **Contact:** {parsed_resume.contact}")
            
            st.markdown("**Professional Summary**")
            st.write(parsed_resume.summary or "Not extracted")

            st.markdown("**Key Skills**")
            if parsed_resume.skills:
                st.write(" • ".join(parsed_resume.skills))
            else:
                st.write("No skills extracted")

            st.markdown("**Work Experience**")
            if parsed_resume.work_experience:
                for exp in parsed_resume.work_experience:
                    st.write(f"• {exp}")
            else:
                st.write("No work experience extracted")

            st.markdown("**Certificates & Education**")
            if parsed_resume.certificates:
                for cert in parsed_resume.certificates:
                    st.write(f"• {cert}")
            else:
                st.write("No certificates extracted")

        st.caption("Powered by LangGraph + Gemini 1.5 Flash • Results are AI-generated suggestions only.")

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        st.info("Common fixes:\n"
                "- Check that your Google API key is valid and has Gemini access\n"
                "- Make sure the PDF is not password-protected or image-only\n"
                "- Try a different resume or shorten very long job descriptions")
    
    finally:
        # Always clean up temp file
        if os.path.exists(resume_path):
            os.unlink(resume_path)

else:
    # Welcome / instructions when no submission yet
    st.info(
        "👆 Fill in the job role, paste the job description, upload your resume (PDF), "
        "then click **Analyze Resume Fit**."
    )

    with st.expander("How it works"):
        st.markdown("""
        1. **Text Extraction** — Your PDF resume is parsed locally.
        2. **Structured Parsing** — Gemini extracts name, skills, experience, etc. into structured data.
        3. **AI Evaluation** — Another Gemini call compares your background against the job role + description.
        4. **Output** — You receive a clear 1–10 rating + detailed, actionable feedback.
        """)

    st.caption("Your resume is processed in-memory and the temporary file is deleted immediately after analysis.")

# Sidebar info
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("""
    **Model:** Gemini 3.1 Flash Lite  
    **Framework:** LangGraph + Streamlit  
    """)
    st.divider()
    st.markdown("Made for rapid resume-job matching. For best results, use detailed job descriptions.")