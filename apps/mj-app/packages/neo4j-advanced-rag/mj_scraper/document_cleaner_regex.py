import os
import re
from concurrent.futures import ThreadPoolExecutor


# Load file names to process and clean
dir_path = os.path.dirname(os.path.abspath("__file__"))
# dir_path = "/Users/vinaytanwer/Desktop/Projects/Chatbots/langchain/apps/mj-app/packages/neo4j-advanced-rag/mj_scraper"

raw_docs_dir = os.path.join(dir_path, "mj_docs/raw")
raw_files = []
for file in os.listdir(raw_docs_dir):
    # check only text files
    if file.endswith('.txt'):
        raw_files.append(file)

raw_files = sorted(raw_files)


# Define the patterns to be removed
patterns = [
    re.compile(r"\s*\* __\s*?Contents x\s*?No matching results found\s*\*\s*\* \* \*\s*__ __\s*", re.MULTILINE),
    re.compile(r"\ *\*  __ Dark\s*__ Light\s*__Contents\s*", re.MULTILINE),
    re.compile(r"\ *\*  __ Dark\s*__ Light\s*\* \* \*\s*Article summary\s*__\s*Did you find this summary helpful\? __ __ __ __\s*__\s*Thank you for your feedback\s*", re.MULTILINE),
    re.compile(r"\s*\* \* \*\s*__\s*Previous\s*?Next\s*__\s*", re.MULTILINE),
    re.compile(r"Table of contents\s*Midjourney is an independent research lab exploring new mediums of thought and\s*expanding the imaginative powers of the human species\. We are a small self-\s*funded team focused on design, human infrastructure, and AI\.\s*FOLLOW US: \[F\] \[T\] \[R\]\s*Support\s*For questions or support visit the  Midjourney Discord support channels\.\s*Sites\s*\* Midjourney Website\s*\* Midjourney Discord\s*__\s*", re.MULTILINE)
]


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

            clean_text = ''.join(raw_text)
            for pattern in patterns:
                clean_text = re.sub(pattern, '', clean_text)

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