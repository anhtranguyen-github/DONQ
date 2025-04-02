from docling.document_converter import DocumentConverter
import re

class DoclingReader:
    def __init__(self):
        self.converter = DocumentConverter()
    
    def read_from_url(self, url):
        """Read and convert document from a URL to markdown"""
        result = self.converter.convert(url)
        return result.document.export_to_markdown()
    
    def read_from_file(self, file_path):
        """Read and convert document from a file path to markdown"""
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()
    
    def read_from_url_as_text(self, url):
        """Read and convert document from a URL to plain text"""
        markdown = self.read_from_url(url)
        text = self._markdown_to_plain_text(markdown)
        return self.normalize_raw_text(text)
    
    def read_from_file_as_text(self, file_path):
        """Read and convert document from a file path to plain text"""
        markdown = self.read_from_file(file_path)
        return self._markdown_to_plain_text(markdown)
    
    def _markdown_to_plain_text(self, markdown):
        """Convert markdown to plain text by removing formatting"""
        # Remove headers (# Header)
        text = re.sub(r'#+\s+(.*)', r'\1', markdown)
        # Remove bold/italic (**text** or *text*)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # Remove code blocks (```code```)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove inline code (`code`)
        text = re.sub(r'`(.*?)`', r'\1', text)
        # Remove bullet points
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        # Remove numbered lists
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Remove horizontal rules (---, ___, ***)
        text = re.sub(r'^\s*[-_*]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove images ![alt](url)
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove links [text](url)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove blockquotes
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text
    
    def normalize_raw_text(self, text):
        """
        Normalize text to make it more compatible with LLM parsers.
        This helps ensure consistent entity extraction.
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence breaks
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)
        
        # Fix common extraction issues
        # Remove any non-breaking spaces
        text = text.replace('\xa0', ' ')
        
        # Remove any strange Unicode characters
        text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F\u0400-\u04FF]+', ' ', text)
        
        # Remove any potential HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        
        # Consolidate sections with a single break
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        # Format into more natural paragraphs
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        for p in paragraphs:
            # Skip empty paragraphs
            if not p.strip():
                continue
            # Format each paragraph
            clean_p = p.replace('\n', ' ').strip()
            formatted_paragraphs.append(clean_p)
        
        # Join with double line breaks for clear paragraph separation
        return '\n\n'.join(formatted_paragraphs)
