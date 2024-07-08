
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup as Soup

url = "https://docs.midjourney.com/docs/"
# url = "https://en.wikipedia.org/wiki/Pedri"

#-----------------------------------------------------------

loader = AsyncChromiumLoader([url])
docs = loader.load()

print(docs[0].page_content)

tt = Html2TextTransformer()

fd = tt.transform_documents(docs)

print(fd[0].page_content)

#-----------------------------------------------------------

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

loader = RecursiveUrlLoader(url=url,
                            max_depth=1000,
                            extractor=lambda x:Soup(x,"html.parser").text)
docs=loader.load()

print(docs[0].page_content)

#-----------------------------------------------------------


loader = AsyncChromiumLoader([url])
docs = loader.load()

bsoup = Soup(docs[0].page_content, "html.parser")

bsoup

# all_links = bsoup.findAll('a')

# for link in all_links:
#     href = link.get('href')
#     try:
#         if href.startswith("/docs"):
#             print(href, flush=True)
#     except:
#         pass

#----------------------------------------------------------

from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.midjourney.com/docs/using-images"

docs = WebBaseLoader(url).load()

docs

#----------------------------------------------------------
import cloudscraper
from bs4 import BeautifulSoup
from langchain_community.document_transformers import Html2TextTransformer

scraper = cloudscraper.create_scraper()

url = "https://docs.midjourney.com/docs/quick-start"
url = "https://docs.midjourney.com/docs/purchase-order-terms-and-conditions"

html = scraper.get(url).text

soup = BeautifulSoup(html, 'html.parser')
# print(soup.get_text(separator='\n'))

from langchain.schema import Document
document = Document(page_content=str(soup), metadata={})

tt = Html2TextTransformer()
fd = tt.transform_documents([document])
print(fd[0].page_content)

#----------------------------------------------------------
import cloudscraper
from bs4 import BeautifulSoup

scraper = cloudscraper.create_scraper()

url = "https://docs.midjourney.com/docs/quick-start"
url = "https://docs.midjourney.com/docs/purchase-order-terms-and-conditions"

html = scraper.get(url).text

soup = BeautifulSoup(html, 'html.parser')
# print(soup.get_text(separator='\n'))

# Replace <br> tags with newline
for br in soup.find_all("br"):
    br.replace_with("\n")

# Replace <p> tags with double newline
for p in soup.find_all("p"):
    p.insert(0, "\n")
    p.insert(0, "\n")

# # Handle <ul> and <ol> tags
# for ul in soup.find_all("ul"):
#     ul.insert(0, "\n")
#     ul.insert(0, "\n")

# for ol in soup.find_all("ol"):
#     ol.insert(0, "\n")
#     ol.insert(0, "\n")

# Handle <li> tags
for li in soup.find_all("li"):
    li.insert(0, "\n- ")
    li.insert(0, "\n")

# Get text with newlines
text = soup.get_text()

print(text)



#----------------------------------------------------------


