import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PyPDF2 import PdfReader
import io 

def load_pdf_from_url(paper_url: str) -> str:

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        # allow_redirects=True is default, but explicit here to show we follow to the PDF
        response = session.get(paper_url, headers=headers, timeout=45, allow_redirects=True)
        
        # Check if we were redirected to the "unsupported_browser" page specifically
        if "unsupported_browser" in response.url:
             raise Exception("ScienceDirect blocked the request (Unsupported Browser). The User-Agent may need further updating.")

        response.raise_for_status()

        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    
    except requests.exceptions.HTTPError as e:
        if response and response.status_code == 403:
            raise Exception("Access to the PDF is forbidden (403). This paper may require institutional authentication.")
        raise Exception(f"HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download PDF: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def load_paper_from_path(paper_path: str) -> str:
    try:
        pdf_reader = PdfReader(paper_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        return f"Error loading uploaded paper: {str(e)}"



