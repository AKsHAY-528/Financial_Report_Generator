
Financial Summary Web App
A simple web application built using Flask, HuggingFace NLP, Langchain, and ReportLab. The app allows users to upload a .txt file containing financial data, generates a summary, and then creates a downloadable PDF report with key financial figures.


## 📂 Project Structure for WebApp

financial_summary_webapp/
│
├── app.py                  # Main Flask application
├── requirements.txt         # List of required Python libraries
│
├── templates/
│   └── upload.html          # Upload page frontend
│
├── uploads/                 # Uploaded financial text files (auto-created, initially empty)
│
├── reports/                 # Generated PDF reports (auto-created, initially empty)
│
└── venv/                    # (Optional) Virtual environment (not uploaded to GitHub)

## 🧩 Financial Summary Script (First Code)

import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load financial data from a text file
file_path = "financial_data.txt"
with open(file_path, "r") as file:
    financial_text = file.read()

# Split text into chunks for efficient retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.create_documents([financial_text])

# Embed documents using HuggingFace (no API key needed)
embeddings = HuggingFaceEmbeddings()
vectors = [embeddings.embed_query(doc.page_content) for doc in docs]

# Create FAISS vector store
dimension = len(vectors[0])  # Get embedding size
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# Query the RAG system
query = "Provide a financial summary for Q4 2024"
query_vector = np.array(embeddings.embed_query(query)).reshape(1, -1)

# Retrieve top 3 relevant chunks
D, I = index.search(query_vector, k=3)
retrieved_text = "\n".join([docs[i].page_content for i in I[0]])

# Generate simple report from retrieved data
report = f"Q4 2024 Financial Summary:\n\n{retrieved_text}"

# Print the generated report
print(report)

## 🧩 Financial Report PDF Generator (Second Code)

import re
import openai
import numpy as np
import faiss
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load financial data from the text file
file_path = "financial_data.txt"
with open(file_path, "r") as file:
    financial_text = file.read()

# AI Model: Use HuggingFace Financial NLP Model for Key Figures Extraction
extractor = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define queries for AI extraction
queries = {
    "Total Revenue": "What is the company's total revenue?",
    "Product Sales Revenue": "How much revenue came from product sales?",
    "Service Revenue": "What is the service revenue?",
    "Subscription Revenue": "How much subscription revenue was earned?",
    "Total Expenses": "What are the total company expenses?",
    "COGS": "What is the cost of goods sold (COGS)?",
    "Marketing & R&D Expenses": "What are the marketing and R&D expenses?",
    "Net Profit": "What is the net profit?",
    "Profit Margin": "What is the company's profit margin percentage?",
    "Future Revenue Projection": "What is the projected revenue for next quarter?",
    "Investment Raised": "How much investment did the company raise?"
}

# Extract financial data using AI
extracted_data = {}
for key, query in queries.items():
    response = extractor(question=query, context=financial_text)
    extracted_data[key] = f"${response['answer']}" if response['score'] > 0.5 else "N/A"

# Generate PDF Report
pdf_file = "Financial_Report_Q4_2024.pdf"

def generate_pdf_report(filename, data):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y_position = height - 50  # Start position

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, y_position, "Financial Report - Q4 2024")
    y_position -= 30

    c.setFont("Helvetica", 12)
    for key, value in data.items():
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, key + ": ")
        y_position -= 20
        c.setFont("Helvetica", 12)
        c.drawString(80, y_position, value)
        y_position -= 30

        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 50

    c.save()

# Generate and save the PDF
generate_pdf_report(pdf_file, extracted_data)

print(f"✅ AI-Powered Financial report successfully saved as {pdf_file}")


🖥️ Web Interface
Upload Page: The user can upload a .txt file containing financial data through the web interface.

PDF Report: After uploading, the app processes the file, extracts relevant financial data, and generates a PDF file.

Download: Users can download the generated PDF report containing the extracted financial details.

🧩 Code Details

Financial Summary Script (First Code)
This script performs financial data embedding and vector-based similarity retrieval using the LangChain library and FAISS for efficient querying.

Financial Report PDF Generator (Second Code)
This script uses HuggingFace NLP for financial data extraction and ReportLab to generate a PDF report with the key figures.


## 🚀 Web App Implementation (Flask)

💡 How the Web App Works
Upload Financial Data: Users upload a .txt file containing financial data via the web interface.

Extract Key Data: Using HuggingFace NLP, the app extracts key financial figures (e.g., revenue, expenses, profit margins).

Generate PDF Report: The app creates a PDF report summarizing the extracted data.

Download the Report: Users can download the generated PDF report directly from the app.
