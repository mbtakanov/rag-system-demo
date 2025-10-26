import os
import random
import requests
from tqdm import tqdm
from time import sleep
from docx import Document
from openai import OpenAI
from PyPDF2 import PdfReader
from datetime import datetime
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from config import DOCX_DIR, PDF_DIR

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os.makedirs(DOCX_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

TOPICS = [
    "AI in Education", "Machine Learning in Retail", "Natural Language Processing in Law",
    "Ethical Data Collection", "Deep Learning in Climate Science", "Human-AI Collaboration",
    "Edge Computing for IoT", "Predictive Maintenance", "Robotics in Manufacturing",
    "AI in Marketing Analytics", "Facial Recognition Regulations", "Smart Cities and AI",
    "Healthcare Predictive Analytics", "Algorithmic Bias in Hiring", "Speech Recognition Trends",
    "AI in Logistics", "Computer Vision in Sports", "Neural Networks in Music",
    "Chatbots in Customer Support", "AI in Agriculture", "AI for Legal Research",
    "AI in Mental Health", "Computer Vision in Retail", "AI for Disaster Management",
    "AI in Sports Analytics", "AI-driven HR Systems", "AI for Wildlife Conservation",
    "AI in Urban Planning", "Voice Recognition in Healthcare", "AI and Cultural Heritage",
    "AI in Manufacturing Efficiency", "AI and Creativity", "AI in Transportation",
    "AI in Marketing Forecasting", "AI for Fraud Detection", "AI in Cybersecurity Training",
    "AI in Language Translation", "AI for Traffic Management", "AI in Customer Experience",
    "AI and Human Behavior", "AI in Predictive Policing", "AI in Drug Development",
    "AI in eCommerce Personalization", "AI in Journalism", "AI for Environmental Monitoring",
    "AI in Retail Supply Chain", "AI for Financial Forecasting", "AI in Energy Efficiency",
    "AI and Data Privacy", "AI in Space Exploration", "AI for Smart Homes",
    "AI for Small Businesses", "AI in Architecture", "AI in Fashion Design",
    "AI in Insurance", "AI in Gaming", "AI in Education Policy", "AI in Music Creation",
    "AI and the Future of Work", "AI in Neuroscience", "AI for Social Good",
    "AI for Accessibility", "AI in Automotive Safety", "AI and Ethics in Law",
    "AI in Art Restoration", "AI in Personalized Medicine", "AI in Climate Adaptation",
    "AI and Governance", "AI in Public Policy", "AI for Scientific Discovery",
    "AI and Creativity in Media", "AI in Construction Management", "AI in Retail Analytics",
    "AI for Mental Wellness", "AI in Government Services"
]

def generate_report(topic: str) -> str:
    # The query can be improved to be more specific, detailed, add tables, etc.
    # (e.g. no meta data, no references, no citations, no summary, no conclusions, no recommendations, etc.)
    
    response = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {"role": "system", "content": "You are a professional research report writer."},
            {"role": "user", "content": f"Write a detailed (at least 5 pages) research report (at least 3000 words) on {topic}."}
        ],
        max_output_tokens=4000,
    )
    return response.output[0].content[0].text

def generate_docx_set(n_docs=50):
    for i in tqdm(range(n_docs)):
        topic = random.choice(TOPICS)
        try:
            text = generate_report(topic)
            doc = Document()

            # TODO: Add metadata and use it later.
            doc.core_properties.subject = topic
            doc.core_properties.created = datetime.now()
            doc.add_heading(f"Report on {topic}", level=1)
            doc.add_paragraph(text)
            doc.save(os.path.join(DOCX_DIR, f"report_{i+1}.docx"))
            sleep(1)
        except Exception as e:
            print(f"Error (DOCX {topic}): {e}")
    print("DOCX generation complete.\n")

def download_arxiv_pdfs(n_pdfs=50, min_pages=5, max_pages=20):
    api_url = "http://export.arxiv.org/api/query"
    
    downloaded = 0
    start = 0
    batch_size = 100
    
    while downloaded < n_pdfs:
        params = {
            "search_query": "cat:cs.ML OR cat:cs.LG OR cat:q-bio OR cat:cs.CR OR cat:cs.CV OR cat:cs.CL",
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        response = requests.get(api_url, params=params)
        root = ET.fromstring(response.content)
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        
        if not entries:
            print("No more results available, retrying next batch...")
            start += batch_size
            continue
        
        for entry in tqdm(entries):
            if downloaded >= n_pdfs:
                break

            pdf_link = entry.find('{http://www.w3.org/2005/Atom}id').text.replace('/abs/', '/pdf/') + '.pdf'
            
            try:
                r = requests.get(pdf_link, timeout=30)
                if r.status_code == 200:
                    temp_path = os.path.join(PDF_DIR, f"temp.pdf")
                    with open(temp_path, "wb") as f:
                        f.write(r.content)
                    
                    reader = PdfReader(temp_path)
                    num_pages = len(reader.pages)
                    
                    if min_pages <= num_pages <= max_pages:
                        final_path = os.path.join(PDF_DIR, f"doc_{downloaded+1}.pdf")
                        os.rename(temp_path, final_path)
                        downloaded += 1
                    else:
                        os.remove(temp_path)
                        print(f"Skipped: {num_pages} pages")
            except Exception as e:
                print(f"Error: {e}")
            
            sleep(3)
        
        start += batch_size
        sleep(3)
    
    print(f"Downloaded {downloaded} PDFs.")

if __name__ == "__main__":
    generate_docx_set()
    download_arxiv_pdfs()
