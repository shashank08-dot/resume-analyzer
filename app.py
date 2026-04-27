import streamlit as st
import PyPDF2
import string
import nltk
import time
import faiss
import spacy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# ------------------ SETUP ------------------

# Download stopwords only once
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Load spacy safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("SpaCy model not installed. Run: python -m spacy download en_core_web_sm")

# Load embedding model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="AI Resume Intelligence System", layout="wide")
st.title("AI Resume Intelligence System")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.lower()

def clean_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join([w for w in text.split() if w not in stop_words])

SECTION_KEYWORDS = {
    "skills": ["skill", "technical skills"],
    "experience": ["experience", "internship"],
    "projects": ["project"],
    "education": ["education"]
}

TECH_SKILLS = [
    "python","java","c++","c","sql","mysql","mongodb","flask","django",
    "aws","docker","html","css","javascript","react","node.js",
    "tensorflow","pytorch","opencv","excel","power bi","tableau"
]

def extract_sections(text):

    sections = {k:"" for k in SECTION_KEYWORDS}
    current = None

    for line in text.split("\n"):

        line_lower = line.lower()

        found = False

        for key,keywords in SECTION_KEYWORDS.items():

            if any(word in line_lower for word in keywords):
                current = key
                found = True
                break

        if not found and current:
            sections[current] += line + " "

    return sections


def chunk_text(text,chunk_size=150,overlap=30):

    words = text.split()
    chunks = []

    for i in range(0,len(words),chunk_size-overlap):

        chunk = " ".join(words[i:i+chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def extract_skills(text):

    text_lower = text.lower()

    skills=set()

    for skill in TECH_SKILLS:

        if skill in text_lower:
            skills.add(skill)

    return skills


def plot_gauge(score,title):

    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=score,

        title={"text":title},

        gauge={"axis":{"range":[0,100]}}

    ))

    st.plotly_chart(fig,use_container_width=True)


# ------------------ ANALYSIS ------------------

if st.button("Analyze"):

    if uploaded_resume and job_description:

        with st.spinner("Analyzing Resume..."):

            time.sleep(0.5)

            resume_text = extract_text_from_pdf(uploaded_resume)

            if not resume_text.strip():
                st.warning("PDF text could not be extracted")
                st.stop()

            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(job_description.lower())

            # -------- SECTION MATCHING --------

            resume_sections = extract_sections(resume_text)

            section_scores = {}

            jd_embedding = model.encode(jd_clean,normalize_embeddings=True).reshape(1,-1)

            for section_name,sec_text in resume_sections.items():

                if not sec_text.strip():
                    section_scores[section_name] = 0
                    continue

                chunks = chunk_text(sec_text)

                embeddings=[]

                for c in chunks:

                    emb = model.encode(c,normalize_embeddings=True)

                    embeddings.append(emb)

                embeddings = np.vstack(embeddings)

                dim = embeddings.shape[1]

                index = faiss.IndexFlatIP(dim)

                index.add(embeddings)

                scores,_ = index.search(jd_embedding,k=len(embeddings))

                section_scores[section_name] = round(float(scores.mean())*100,2)


            overall_match = round(np.mean(list(section_scores.values())),2)

            # -------- SKILL ANALYSIS --------

            resume_skills = extract_skills(resume_text)

            jd_skills = extract_skills(job_description)

            matched_skills = sorted(resume_skills & jd_skills)

            missing_skills = sorted(jd_skills - resume_skills)


        # -------- DISPLAY RESULTS --------

        st.subheader("Overall Resume Match")

        plot_gauge(overall_match,"Overall Resume-JD Match (%)")

        st.subheader("Section Match")

        for sec,score in section_scores.items():

            plot_gauge(score,f"{sec.capitalize()} Match (%)")

        col1,col2 = st.columns(2)

        with col1:

            st.subheader("Matching Skills")

            if matched_skills:

                st.table(pd.DataFrame(matched_skills,columns=["Skill"]))

            else:

                st.write("None")


        with col2:

            st.subheader("Missing Skills")

            if missing_skills:

                st.table(pd.DataFrame(missing_skills,columns=["Skill"]))

                st.write("Recommendation: Add these skills if relevant.")

            else:

                st.write("No missing skills found.")


        st.subheader("Insight")

        st.write(
            "Resume matching uses semantic similarity with Sentence Transformers "
            "and FAISS vector search. Section scores help identify weak areas."
        )

    else:

        st.warning("Please upload resume and job description") 


