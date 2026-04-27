# AI Resume Intelligence System

## Overview

AI Resume Intelligence System is an AI-powered web application that analyzes a candidate’s resume against a job description and calculates how well they match.

The system extracts text from uploaded resumes, performs semantic similarity analysis using embeddings and vector search, identifies matching and missing skills, computes section-wise and overall match percentages, and visualizes the results in an interactive dashboard.

This project helps automate resume screening and provides recruiter-style insights.

---

## Features

- Resume PDF text extraction
- Job Description matching
- Semantic similarity scoring
- Section-wise analysis:
  - Skills
  - Experience
  - Projects
  - Education
- Matching skills detection
- Missing skills identification
- Overall Resume–JD Match %
- Interactive dashboard visualizations
- Skill gap recommendations

---

## Tech Stack

### Programming Language
- Python

### Frontend / UI
- Streamlit

### Libraries and Tools
- PyPDF2
- NLTK
- SpaCy
- Sentence Transformers
- FAISS
- NumPy
- Pandas
- Plotly

---

## Project Architecture

```text
Resume PDF + Job Description
            |
            v
    Text Extraction (PyPDF2)
            |
            v
 Text Cleaning (NLTK Stopwords)
            |
            v
 Section Extraction
            |
            v
Text Chunking + Embeddings
(Sentence Transformers)
            |
            v
Vector Similarity Search
(FAISS)
            |
            v
Match Score Calculation
            |
            v
Skill Gap Analysis
            |
            v
Dashboard Visualization
(Streamlit + Plotly)
```

---

## How It Works

### 1. Resume Parsing
The uploaded PDF resume is processed and text is extracted page by page.

### 2. Text Preprocessing
The text is cleaned by:
- Lowercasing
- Removing punctuation
- Removing stopwords

### 3. Section Extraction
The system detects sections such as:
- Skills
- Experience
- Projects
- Education

### 4. Semantic Embeddings
Resume chunks and job descriptions are converted into embeddings using:

all-MiniLM-L6-v2

### 5. Similarity Search
FAISS performs vector similarity search to compare resume content against the job description.

### 6. Match Score Calculation

Section score:

Score = Mean Similarity × 100

Overall match:

Overall Match = Average of All Section Scores

### 7. Skill Analysis
The system identifies:
- Matching skills
- Missing skills

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/ai-resume-intelligence-system.git
cd ai-resume-intelligence-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download SpaCy model:

```bash
python -m spacy download en_core_web_sm
```

Run the app:

```bash
streamlit run app.py
```

---

## Requirements

Create requirements.txt:

```text
streamlit
PyPDF2
nltk
faiss-cpu
spacy
numpy
pandas
plotly
sentence-transformers
```

---

## Example Output

The system generates:

- Overall Resume Match %
- Section-wise Match Scores
- Matching Skills Table
- Missing Skills Table
- Recommendations

Example:

```text
Overall Match: 82%

Skills Match: 85%
Experience Match: 76%
Projects Match: 90%
Education Match: 80%

Missing Skills:
- Docker
- AWS
- React
```

---

## Challenges Solved

### Resume Formatting Variations
Handled inconsistent PDF structures using section extraction logic.

### Keyword Matching Limitations
Solved using semantic embeddings.

### Fast Similarity Search
Implemented FAISS vector indexing for efficient comparisons.

---

## Future Enhancements

- Resume ranking for multiple candidates
- ATS compatibility scoring
- LLM-based resume improvement suggestions
- Cloud deployment using AWS
- Advanced skill extraction using SpaCy NER

---

## Use Cases

- Recruiters for initial screening
- Students checking resume-job fit
- Job seekers identifying missing skills
- HR teams automating candidate evaluation

---

## Learning Outcomes

Through this project I gained practical experience in:

- NLP preprocessing
- Semantic embeddings
- Vector databases / similarity search
- Streamlit app development
- Data visualization
- AI-powered application design

---

## Project Demo

Add screenshots here:

```text
screenshots/dashboard.png
screenshots/skills_analysis.png
```

(Optional: add deployed Streamlit link)

---

## Author

Shashank K R

LinkedIn: https://linkedin.com/in/shashank-kr-ai

GitHub: https://github.com/Shashank08-dot

---

## License

This project is for educational and portfolio purposes.