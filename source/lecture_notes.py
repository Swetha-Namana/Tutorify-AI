import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import logging

from dotenv import load_dotenv

load_dotenv()


def download_lecture_notes(save_dir=os.getenv("PDF_DIRECTORY"), url="https://inst.eecs.berkeley.edu/~cs188/su24/"):
    """Download lecture notes from the specified URL and save them to the directory."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create the directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create directory '{save_dir}': {e}")
        return

    # Attempt to retrieve the webpage
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve URL '{url}': {e}")
        return

    # Parse the webpage content
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        logging.error(f"Failed to parse content from '{url}': {e}")
        return

    # Extract PDF links
    try:
        links = soup.find_all('a', href=True)
        pdf_links = [urljoin(url, link['href']) for link in links if link['href'].endswith('.pdf')]
        if not pdf_links:
            logging.warning("No PDF links found on the page.")
            return
    except Exception as e:
        logging.error(f"Error extracting PDF links: {e}")
        return

    # Download each PDF file
    for pdf_link in pdf_links:
        try:
            pdf_response = requests.get(pdf_link, timeout=10)
            pdf_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download '{pdf_link}': {e}")
            continue  # Proceed to the next file

        file_name = os.path.join(save_dir, os.path.basename(pdf_link))

        # Save the PDF file
        try:
            with open(file_name, 'wb') as file:
                file.write(pdf_response.content)
            logging.info(f"Downloaded: {file_name}")
        except Exception as e:
            logging.error(f"Failed to save file '{file_name}': {e}")


# Only call the function when running this script directly, not when imported
if __name__ == "__main__":
    download_lecture_notes()
