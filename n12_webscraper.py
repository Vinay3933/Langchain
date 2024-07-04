# from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import AsyncChromiumLoader
# from langchain_community.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup as Soup

url = "https://docs.midjourney.com/docs/"
# url = "https://en.wikipedia.org/wiki/Pedri"

#-----------------------------------------------------------

# loader = AsyncChromiumLoader([url])
# docs = loader.load()

# print(docs[0].page_content)

# tt = Html2TextTransformer()

# fd = tt.transform_documents(docs)

# print(fd[0].page_content)

#-----------------------------------------------------------

# loader = RecursiveUrlLoader(url=url,
#                             max_depth=1000,
#                             extractor=lambda x:Soup(x,"html.parser").text)
# docs=loader.load()

# print(docs[0].page_content)

#-----------------------------------------------------------


loader = AsyncChromiumLoader([url])
docs = loader.load()

bsoup = Soup(docs[0].page_content, "html.parser")

all_links = bsoup.findAll('a')

for link in all_links:
    href = link.get('href')
    try:
        if href.startswith("/docs"):
            print(href, flush=True)
    except:
        pass



