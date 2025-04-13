import requests
from bs4 import BeautifulSoup as BS
import lxml
import csv
import re

URL="https://millercenter.org/the-presidency/presidential-speeches"
HEADER = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}
OUTPUT_DIR = "data_scrape"
TXT_DIR = "data_scrape/txts"

def give_simple_filename(text: str) -> str:
    name = text.strip()
    name = name.lower()
    name = re.sub(r'[\\/*?:"<>| ]+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name

def scrape_transcript(title: str, url: str) -> str:
    url = url+"#dp-expandable-text"
    resp = requests.get(url, headers=HEADER)
    soup = BS(resp.content, "lxml")
    transcript_text = soup.find("div", {"id": "dp-expandable-text"}).text
    transcript_text = transcript_text.strip()
    name = give_simple_filename(title)
    with open(f"{TXT_DIR}/{name}.txt", "w") as f:
        f.write(transcript_text)
    return name


# Scraping title and url from the website
resp = requests.get(URL, headers=HEADER)
soup = BS(resp.content, "lxml")
all_transcripts = soup.find_all("div", {"class":"views-row"})
for transcript in all_transcripts:
    transcript_data = transcript.find('a', href=True)
    speech_title = transcript_data.text
    transcript_url = transcript_data['href']
    print(speech_title, transcript_url)
    filename = scrape_transcript(speech_title, transcript_url)
    # Save the data in a CSV
    with open(f"{OUTPUT_DIR}/title_and_links.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([transcript_data.text, transcript_data['href'], filename])
