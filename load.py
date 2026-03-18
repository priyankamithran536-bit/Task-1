import pdfplumber
from PIL import Image
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_documents(data_folder):
    docs = []

    for file in os.listdir(data_folder):
        path = os.path.join(data_folder, file)

        if file.endswith(".pdf"):
            text = extract_text_from_pdf(path)
            docs.append({"type": "text", "content": text, "source": file})

        elif file.endswith((".png", ".jpg", ".jpeg")):
            docs.append({"type": "image", "content": path, "source": file})

    return docs