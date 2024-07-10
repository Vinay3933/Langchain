import os
from bs4 import BeautifulSoup
import cloudscraper
from langchain_community.document_transformers import Html2TextTransformer
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document


# Load the list of Routes to scrape
mj_host = "https://docs.midjourney.com"

dir_path = os.path.dirname(os.path.abspath("__file__"))
# dir_path = "/Users/vinaytanwer/Desktop/Projects/Chatbots/langchain/apps/mj-app/packages/neo4j-advanced-rag/mj_scraper/"

filename = "midjourney docs links.txt"
mj_docs_routes_file = os.path.join(dir_path, filename)


with open(mj_docs_routes_file, "r") as f:
    mj_routes = f.readlines()

routes_list = [line.strip() for line in mj_routes]

# Scrape the URLs
scraper = cloudscraper.create_scraper()
tt = Html2TextTransformer()

def scrape_and_save(route):
    try:
        url = mj_host + route
        file_name = route.replace("/docs/", "").replace("/", "-") + ".txt"
        file_path = os.path.join(dir_path, "mj_docs/raw/", file_name)

        if not os.path.exists(file_path):
            html = scraper.get(url).text
            soup = BeautifulSoup(html, 'html.parser')
            
            document = Document(page_content=str(soup), metadata={})
            fd = tt.transform_documents([document])

            with open(file_path, "w") as f:
                f.write(fd[0].page_content)
            
            print(f"Successfully scraped and saved: {file_name}")
        else:
            print(f"File already exists: {file_name}")

    except Exception as e:
        print(f"Failed to scrape {route}: {e}")

# Use ThreadPoolExecutor to scrape concurrently
with ThreadPoolExecutor() as executor:
    executor.map(scrape_and_save, routes_list)