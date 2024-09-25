import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

# Directory to save the PDFs
save_dir = "lecture notes"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

url = "https://inst.eecs.berkeley.edu/~cs188/su24/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract and download lecture notes (hypothetical, adapt based on site structure)
links = soup.find_all('a', href=True)
pdf_links = [urljoin(url, link['href']) for link in links if link['href'].endswith('.pdf')]

# Download each PDF file
for pdf_link in pdf_links:
    response = requests.get(pdf_link)
    file_name = os.path.join(save_dir, pdf_link.split('/')[-1])
    with open(file_name, 'wb') as file:
        file.write(response.content)

    print(f"Downloaded: {file_name}")
