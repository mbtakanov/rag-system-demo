import os
import sys
from tqdm import tqdm
from pathlib import Path

from docling.document_converter import PdfFormatOption
from docling.document_converter import DocumentConverter, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, smolvlm_picture_description

import config

RAW_DATA_DIR = config.RAW_DATA_DIR
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# If we need cutting-edge accuracy, we can use Google Cloud Vision, Amazon Textract, etc.
pdf_pipeline_options = PdfPipelineOptions(
    generate_page_images=True, 
    do_ocr=True,
    do_picture_description=True,
    ocr_options=RapidOcrOptions(),
    picture_description_options=smolvlm_picture_description
)

pdf_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)}
)

docx_converter = DocumentConverter()


def extract_pdf_with_docling(file_path: str) -> str:
    try:
        result = pdf_converter.convert(Path(file_path))
        return result.document.export_to_markdown()
    except Exception as e:
        print(f"[ERROR] PDF extraction failed for {file_path}: {e}")
        return ""


def extract_docx_with_docling(file_path: str) -> str:
    try:
        result = docx_converter.convert(Path(file_path))
        return result.document.export_to_markdown()
    except Exception as e:
        print(f"[ERROR] DOCX extraction failed for {file_path}: {e}")
        return ""


def process_all_documents(raw_dir: str = RAW_DATA_DIR):
    """Process all documents from raw_dir, expecting pdf/ and docx/ subdirectories.
    
    Args:
        raw_dir: Path to the raw documents directory (default: uses global RAW_DIR)
    """
    for subfolder in ["pdf", "docx"]:
        folder = os.path.join(raw_dir, subfolder)
        if not os.path.exists(folder):
            print(f"[WARN] Directory {folder} does not exist, skipping...")
            continue

        print(f"\nProcessing {subfolder.upper()} files from {folder}...")
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.pdf', '.docx'))]

        if not files:
            print(f"[WARN] No {subfolder.upper()} files found in {folder}")
            continue

        for filename in tqdm(files, desc=f"{subfolder.upper()}"):
            path = os.path.join(folder, filename)

            if filename.lower().endswith(".pdf"):
                text = extract_pdf_with_docling(path)
            elif filename.lower().endswith(".docx"):
                text = extract_docx_with_docling(path)
            else:
                continue

            if not text.strip():
                print(f"[WARN] No text extracted from {filename}")
                continue

            out_path = os.path.join(
                PROCESSED_DATA_DIR,
                filename.replace(".pdf", "_pdf.md").replace(".docx", "_docx.md")
            )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

    print("\nPreprocessing complete. Clean text saved in data/processed/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        process_all_documents(raw_dir=source_dir)
    else:
        process_all_documents()
