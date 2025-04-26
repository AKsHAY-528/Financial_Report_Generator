from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load Huggingface Model only once
extractor = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Financial Queries
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

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/generate', methods=['POST'])
def generate_report():
    file = request.files['financial_file']
    
    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read uploaded text
    with open(filepath, 'r', encoding='utf-8') as f:
        financial_text = f.read()

    # (Optional) Split text if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents([financial_text])

    # For now, combine all chunks again (you can improve later to handle large documents)
    full_text = " ".join(doc.page_content for doc in docs)

    # Extract Financial Data
    extracted_data = {}
    for key, question in queries.items():
        response = extractor(question=question, context=full_text)
        if response['score'] > 0.5:
            extracted_data[key] = f"${response['answer']}"
        else:
            extracted_data[key] = "N/A"

    # Generate PDF Report
    pdf_filename = f"{file.filename.split('.')[0]}_Financial_Report_Q4_2024.pdf"
    pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y_position = height - 50  # Start position

    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, y_position, "Financial Report - Q4 2024")
    y_position -= 40  # Move down

    c.setFont("Helvetica", 12)
    for key, value in extracted_data.items():
        if y_position < 100:  # Start a new page if too low
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 50
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, key + ":")
        y_position -= 20
        c.setFont("Helvetica", 12)
        c.drawString(70, y_position, value)
        y_position -= 30

    c.save()

    # Send the file to user
    return send_from_directory(REPORT_FOLDER, pdf_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
