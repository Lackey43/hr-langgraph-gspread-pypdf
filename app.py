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

from tools.google_sheets import append_to_google_sheet


class PdfExtractor(BaseModel):
    """Structured extraction from resume PDF."""

    name: str = Field(description="Full name of the applicant")
    age: str = Field(description="Age of the applicant (if mentioned)")
    contact: str = Field(description="Phone number or contact details")
    email: str = Field(description="Email address of the applicant")
    summary: str = Field(description="Professional summary / overall profile overview")
    skills: list[str] = Field(description="List of technical and soft skills")
    work_experience: list[str] = Field(
        description="List of work experiences with company, role, and key achievements"
    )
    certificates: list[str] = Field(
        description="List of relevant certificates, courses, or licenses"
    )


class Recommendation(BaseModel):
    """Final rating and feedback."""

    rate: int = Field(
        description="Rating from 1 to 10 indicating how qualified the applicant is for the role"
    )
    recom: str = Field(
        description="Detailed assessment, strengths, gaps, and actionable feedback"
    )


class State(TypedDict):
    """LangGraph state."""

    job_role: str
    job_description: str
    resume_path: str
    extracted: str
    output: Annotated[list, operator.add]


def extract_pdf(file_path: str) -> str:
    """Extract clean text from PDF resume."""
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()
    text_content = [page.page_content for page in pages]
    return "\n\n".join(text_content)


def extract_data(state: State) -> dict:

    extracted_text = extract_pdf(state["resume_path"])
    return {"extracted": extracted_text}


def parse_pdf(state: State) -> dict:

    message = structured_ai.invoke(
        [
            SystemMessage(
                "You are a precise resume parser. Extract name, age, contact info, email, "
                "professional summary, skills, work experience, and certificates from the provided resume text."
            ),
            HumanMessage(f"Resume text:\n{state['extracted']}"),
        ]
    )
    skills = "\n".join(message.skills)
    wokrexp = "\n".join(message.work_experience)
    certs = "\n".join(message.certificates)
    print(skills)
    data = [
        message.name,
        message.email,
        message.contact,
        message.age,
        wokrexp,
        skills,
        certs,
    ]
    print(data)
    append_to_google_sheet(column="A", values=data)
    return {"output": [message]}


def show_result(state: State) -> dict:

    job_role = state["job_role"]
    job_desc = state.get("job_description", "")
    applicant_data = (
        state["output"][0] if state.get("output") else "No structured data available"
    )

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
        [SystemMessage(system_prompt), HumanMessage(human_prompt)]
    )

    data = [message.rate, message.recom, state["job_role"], state["job_description"]]

    print(data)
    append_to_google_sheet(column="H", values=data)
    return {"output": [message]}


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    google_model = None
    structured_ai = None
    structured_ai2 = None
    agent = None
else:
    google_model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", api_key=api_key, temperature=1.0
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


st.set_page_config(
    page_title="Resume Fit Analyzer",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("HR AI Resume Fit Analyzer")
st.markdown(
    "Upload your resume (PDF), enter the job role + description, and get an instant "
    "**rating (1-10)** + professional feedback powered by Gemini + LangGraph.\n\n**Demo by Claude Daigan**"
    "\n\nCheck your output in [Google Sheets](https://docs.google.com/spreadsheets/d/1OsGCgyeG76Og5sORQTi4N3ha7nxmYKnzqleaO5o13C4/edit?gid=469343244#gid=469343244)!"
)

st.divider()


with st.form(key="analyzer_form", clear_on_submit=False):
    col1, col2 = st.columns([1, 1])

    with col1:
        job_role = st.text_input(
            "Job Role / Position Title",
            placeholder="e.g. Senior Full Stack Engineer",
            help="The exact title of the position you're applying for",
        )

    with col2:
        st.write("")

    job_description = st.text_area(
        "Job Description",
        placeholder="Paste the full job description here (requirements, responsibilities, qualifications...)",
        height=180,
        help="The more detailed the job description, the better the analysis",
    )

    uploaded_file = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=["pdf"],
        help="Your resume is processed locally and never stored permanently.",
        label_visibility="visible",
    )

    submitted = st.form_submit_button(
        "🚀 Analyze Resume Fit", use_container_width=True, type="primary"
    )


if submitted:
    if not api_key:
        st.error(
            "❌ GOOGLE_API_KEY not found. Please create a `.env` file with your Google API key."
        )
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

            result = agent.invoke(initial_state)

            parsed_resume: PdfExtractor = result["output"][0]
            recommendation: Recommendation = result["output"][1]

            status.update(
                label="✅ Analysis complete!", state="complete", expanded=False
            )

        st.success("Analysis finished successfully!")

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
                delta_color=delta_color,
            )
        with col_r2:
            st.progress(rating / 10.0, text=f"Fit Score: {rating} / 10")

        st.subheader("💡 AI Feedback & Recommendations")
        st.markdown(recommendation.recom)

        with st.expander(
            "📄 View Parsed Resume Data (for transparency)", expanded=False
        ):
            st.markdown(f"**Name:** {parsed_resume.name}")
            st.markdown(
                f"**Age:** {parsed_resume.age} &nbsp;&nbsp;|&nbsp;&nbsp; **Email:** {parsed_resume.email} &nbsp;&nbsp;|&nbsp;&nbsp; **Contact:** {parsed_resume.contact}"
            )

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

        st.caption(
            "Powered by LangGraph + Gemini 3.1 Flash Lite • Results are AI-generated suggestions only."
        )

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        st.info(
            "Common fixes:\n"
            "- Check that your Google API key is valid and has Gemini access\n"
            "- Make sure the PDF is not password-protected or image-only\n"
            "- Try a different resume or shorten very long job descriptions"
        )

    finally:
        if os.path.exists(resume_path):
            os.unlink(resume_path)

else:
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

    st.caption(
        "Your resume is processed in-memory and the temporary file is deleted immediately after analysis."
    )


with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("""
    **Model:** Gemini 3.1 Flash Lite  
    **Framework:** LangGraph + Streamlit  
    """)
    st.divider()
    st.markdown(
        "Made for rapid resume-job matching. For best results, use detailed job descriptions."
    )
