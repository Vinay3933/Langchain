import os
from concurrent.futures import ThreadPoolExecutor
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load file names to process and clean
dir_path = os.path.dirname(os.path.abspath("__file__"))
dir_path = "/Users/vinaytanwer/Desktop/Projects/Chatbots/langchain/apps/mj-app/packages/neo4j-advanced-rag/mj_scraper"

raw_docs_dir = os.path.join(dir_path, "mj_docs/raw")
raw_files = []
for file in os.listdir(raw_docs_dir):
    # check only text files
    if file.endswith('.txt'):
        raw_files.append(file)

raw_files = sorted(raw_files)


# Load the Model, define the Prompt template and create the Chain

local_llm = "llama3:70b"

llm = ChatOllama(model=local_llm, 
                #  format="json", 
                 temperature=0)

prompt = PromptTemplate(
template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI language model designed to process and clean scraped documentation from the Midjourney website. 
Your task is to identify and remove specific repeated content from each webpage provided by user, ensuring that the remaining text remains intact and unmodified. 
Below are the 5 patterns of the repeated content that need to be removed:

Repeated Content Patterns provided in <pattern><</pattern> XML tag below:

<pattern>
  * __

Contents x

No matching results found

  * 

* * *

__ __
</pattern>

<pattern>
  *  __ Dark

 __ Light

 __Contents
</pattern>

<pattern>
  *  __ Dark

 __ Light

* * *

Article summary

 __

Did you find this summary helpful? __ __ __ __

__

Thank you for your feedback
</pattern>

<pattern>
* * *

 __

Previous

Next

 __
</pattern>

<pattern>
Table of contents

Midjourney is an independent research lab exploring new mediums of thought and
expanding the imaginative powers of the human species. We are a small self-
funded team focused on design, human infrastructure, and AI.

FOLLOW US: [F] [T] [R]

Support

For questions or support visit the  Midjourney Discord support channels.

Sites

  * Midjourney Website
  * Midjourney Discord

__
</pattern>

When processing the text, ensure that all instances of the above patterns are completely removed. The rest of the text should remain unchanged.
Do not include any preamble statement before the final answer. Your final answer should only include the clean text. 
Refrain from adding any preamble statement in your answer.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question:
{text}  
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=['text']
)

chain = prompt | llm | StrOutputParser()

# Directory to save processed files
final_docs_dir = os.path.join(dir_path, "mj_docs/final")

# Function to clean the text file and save under final directory
def clean_and_save(textFile):
    try:
        raw_file = os.path.join(raw_docs_dir, textFile)
        output_file = os.path.join(final_docs_dir, textFile)

        if (not os.path.exists(output_file)) and (os.path.exists(raw_file)):
            with open(raw_file, "r") as f:  
                raw_text = f.readlines()

            clean_text = chain.invoke({"text":raw_text})

            with open(output_file, "w") as f:
                f.write(clean_text)
            
            print(f"Successfully processed and saved: {textFile}")
        else:
            if os.path.exists(output_file):
                print(f"Clean file already exists: {textFile}")
            else:
                print(f"Raw file does't exist: {textFile}")

    except Exception as e:
        print(f"Failed to process {textFile}: {e}")

# Use ThreadPoolExecutor to scrape concurrently
with ThreadPoolExecutor() as executor:
    executor.map(clean_and_save, raw_files)