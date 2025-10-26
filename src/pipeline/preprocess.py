"""
Document preprocessing module.
Converts documents to markdown format using Docling.
"""

import os
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    smolvlm_picture_description
)
from docling.document_converter import PdfFormatOption


class DocumentPreprocessor:
    def __init__(self, 
                 use_ocr: bool = True,
                 use_picture_description: bool = True,
                 use_vlm: bool = False):
        """
        Args:
            use_ocr: Enable OCR for scanned documents
            use_picture_description: Generate descriptions for images
            use_vlm: Use Vision-Language Model for advanced image understanding
        """
        self.use_ocr = use_ocr
        self.use_picture_description = use_picture_description
        self.use_vlm = use_vlm
        
        self.pdf_pipeline_options = self._configure_pdf_pipeline()
        
        self.pdf_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pdf_pipeline_options)
            }
        )
        self.docx_converter = DocumentConverter()

    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str,
                         file_extensions: tuple = ('.pdf', '.docx')) -> dict:
        """
        Args:
            input_dir: Directory containing raw documents
            output_dir: Directory to save processed markdown files
            file_extensions: Tuple of supported file extensions
            
        Returns:
            Dictionary with processing statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        files = [f for f in os.listdir(input_dir) 
                if f.lower().endswith(file_extensions)]
        
        for filename in files:
            stats['total'] += 1
            input_path = os.path.join(input_dir, filename)
            
            text = self._extract(input_path)
            
            if not text or not text.strip():
                print(f"[WARN] No text extracted from {filename}")
                stats['failed'] += 1
                continue
            
            output_filename = Path(filename).stem + '.md'
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                stats['successful'] += 1
            except Exception as e:
                print(f"[ERROR] Failed to save {output_filename}: {e}")
                stats['failed'] += 1
        
        return stats

    def _configure_pdf_pipeline(self) -> PdfPipelineOptions:
        options = PdfPipelineOptions(
            generate_page_images=self.use_picture_description or self.use_vlm,
            do_ocr=self.use_ocr,
            do_picture_description=self.use_picture_description,
        )
        
        if self.use_ocr:
            options.ocr_options = RapidOcrOptions()
        
        if self.use_picture_description and self.use_vlm:
            options.picture_description_options = smolvlm_picture_description
            options.picture_description_options.prompt = (
                "Describe the image in three sentences. Be concise and accurate."
            )
        
        return options
    
    def _extract_pdf(self, file_path: str) -> Optional[str]:
        try:
            result = self.pdf_converter.convert(Path(file_path))
            markdown_text = result.document.export_to_markdown()
            return self._clean_text(markdown_text)
        except Exception as e:
            print(f"[ERROR] PDF extraction failed for {file_path}: {e}")
            return None
    
    def _extract_docx(self, file_path: str) -> Optional[str]:
        try:
            result = self.docx_converter.convert(Path(file_path))
            markdown_text = result.document.export_to_markdown()
            return self._clean_text(markdown_text)
        except Exception as e:
            print(f"[ERROR] DOCX extraction failed for {file_path}: {e}")
            return None
    
    def _extract(self, file_path: str) -> Optional[str]:
        extension = Path(file_path).suffix.lower()
        
        if extension == '.pdf':
            return self._extract_pdf(file_path)
        elif extension == '.docx':
            return self._extract_docx(file_path)
        else:
            print(f"[WARN] Unsupported file format: {extension}")
            return None
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = "\n".join(line.strip() for line in text.split("\n"))
        
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        
        return text.strip()
    