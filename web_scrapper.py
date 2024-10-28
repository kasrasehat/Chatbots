import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time


class WebDataExtractor:
    def __init__(self, base_url, max_depth=1):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited_urls = set()
        self.extracted_data = []

    def is_valid_url(self, url):
        # Check if URL is in the same domain as the base URL
        parsed_base_url = urlparse(self.base_url)
        parsed_url = urlparse(url)
        return parsed_url.netloc == parsed_base_url.netloc

    def extract_content(self, url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                content = {
                    'url': url,
                    'title': soup.title.string if soup.title else 'No Title',
                    'text': " ".join([p.get_text() for p in soup.find_all('p')]),
                    'headers': [header.get_text() for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
                    'images': [img['src'] for img in soup.find_all('img') if img.get('src')],
                }
                return content
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
        return None

    def crawl(self, url, depth=0):
        if url in self.visited_urls or depth > self.max_depth:
            return
        self.visited_urls.add(url)

        print(f"Crawling: {url} at depth {depth}")
        content = self.extract_content(url)
        if content:
            self.extracted_data.append(content)

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    absolute_url = urljoin(url, link['href'])
                    if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                        self.crawl(absolute_url, depth + 1)
        except requests.RequestException as e:
            print(f"Error crawling {url}: {e}")

    def start_crawling(self):
        self.crawl(self.base_url)

    def get_data(self):
        return self.extracted_data


# Usage
if __name__ == "__main__":
    main_url = "https://www.kindermann.de/en/"  # Replace with any website
    extractor = WebDataExtractor(main_url, max_depth=1)
    extractor.start_crawling()

    # Output the collected data
    for data in extractor.get_data():
        print(data)
