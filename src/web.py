
import os
from typing import List
import requests
import time

def fetch_wikipedia_summary(keyword, lang='en', verbose=False):
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{keyword}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if verbose:
            print("Found page:", data.get("title"))
            print("Summary:", data.get("extract"))
            print("URL:", data.get("content_urls", {}).get("desktop", {}).get("page"))
            print()
        return data.get('title')
    else:
        print("Failed to fetch data:", response.status_code)
        return None

def process_wikipedia_content(content: str) -> str:
    '''Remove one character lines and empty lines.'''
    content = content.replace('\n\n', '\n')
    content_lines = content.split('\n')
    content_lines = [line.strip() for line in content_lines]
    content_lines = [line for line in content_lines if line != '' and line != '\n' and len(line) > 1]
    content = '\n'.join(content_lines)
    return content

def fetch_wikipedia_content(keyword, lang='en'):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": keyword
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        pages = response.json().get("query", {}).get("pages", {})

        if len(pages) == 0:
            print("No pages found.")
            return None
        
        page = list(pages.values())[0]
        return process_wikipedia_content(page.get("extract"))
    else:
        print("Failed to fetch data:", response.status_code)
        return None

def convert_keyword_to_valid_filename(keyword: str) -> str:
    '''Converts a keyword to a valid filename by replacing spaces with underscores. and removing special characters.'''
    import re
    valid_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
    keyword = keyword.strip()
    keyword = keyword.replace(' ', '_')
    keyword = re.sub(r'[^' + valid_chars + ']', '', keyword)
    keyword = keyword.lower()
    return keyword

def download_web_pages_by_keywords(keywords: List[str], out_dir: str = None):
    if out_dir is None:
        raise ValueError("out_dir must be specified.")
    
    for keyword in keywords:
        valid_file_name = convert_keyword_to_valid_filename(keyword)
        out_file_path = os.path.join(out_dir, valid_file_name + '.txt')
        if os.path.exists(out_file_path):
            print(f"-- Skipping `{keyword}` because it already exists.")
            continue

        # print(f"Looking for keyword `{keyword}`...")
        title = fetch_wikipedia_summary(keyword)
        content = fetch_wikipedia_content(title)
        time.sleep(2)
        if title is None or content is None:
            print( f"-- Skipping `{keyword}` because it could not be found on Wikipedia.")
            continue
        
        with open(out_file_path, 'w', encoding="utf8") as f:
            f.write(title)
            f.write('\n\n')
            f.write(content)
        print(f"-- Saved {keyword} to {out_file_path}\n")
    
