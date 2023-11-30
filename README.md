# Conversational Search and Question Answering System
This project demonstrates a conversational search and question-answering system using natural language processing (NLP) techniques and machine learning models. It enables users to interact with a system that can retrieve relevant information and provide comprehensive answers based on their queries.

## System Overview
The system comprises several components working together to facilitate conversational search and question answering:

1. **Text Preprocessing and Embeddings:** Text data is processed and transformed into vector representations using language models.

2. **Vector Storage and Retrieval:** Vectorized data is efficiently stored and retrieved using a vector store, allowing for fast similarity searching.

3. **Conversational Retrieval:** Relevant information is retrieved from the vector store based on the user's query.

4. **Question Answering:** The retrieved information is used to answer the user's question using a large language model (LLM).

## Code Sections

The code is organized into distinct sections, each contributing to the system's functionality.

### 1. Installing Required Packages
```python
!pip install -q langchain
!pip install -q replicate
!pip install -q chromadb
!pip install -q pypdf
!pip install -q sentence-transformers
```

This section installs the necessary Python packages using pip. The packages include langchain for NLP, replicate for model replication, chromadb for vector storage, pypdf for working with PDFs, and sentence-transformers for text embeddings.

### 2. Importing Required Modules
```python
import os, sys, csv

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
```

This section imports the necessary modules and classes for text processing, embeddings, vector storage, conversation handling, and document storage. It includes components for both conversational retrieval and question answering.

### 3. Setting Replicate API Token
```python
os.environ["REPLICATE_API_TOKEN"] = "r8_SG8aoSUQkMOP91vcxYbd00YLv64rHMg1En6Qy"
```

Here, the replicate API token is set as an environment variable. Replicate is a platform that facilitates model replication, and the API token is necessary for authentication.

### 4. Reading and Preparing Data
```python
def read_data(csv_file_path):
  # Function to read the CSV and return the text to embed
  # ...

data = read_data("bigBasketProducts.csv")
```

This section defines a function to read data from a CSV file and prepares it for embedding. The data is then read from the "bigBasketProducts.csv" file using this function.

**Suggestion:** To include chat history in the data, modify the `read_data` function to incorporate chat history alongside other information.

### 5. Printing Data Example
```python
print(data[0])
```

This section prints an example document from the prepared data. It provides a glimpse of the data structure and content.

### 6. Initializing Hugging Face Embeddings
```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
```

This section initializes Hugging Face embeddings using the 'sentence-transformers/all-MiniLM-L6-v2' model, which will be used for converting text into vector embeddings.

### 7. Creating and Persisting Chroma Vector Store
```python
db2 = Chroma.from_documents(data, embeddings, persist_directory="./chroma_db")
db2.persist()

my_retriever = db2.as_retriever(search_kwargs={'k':2})
```

Here, a chroma vector store is created from the prepared data using the specified embeddings. The vector store is then persisted to disk for efficient storage and retrieval. A retriever is also created for searching vectors with specified parameters.

### 8. Performing Similarity Search
```python
query = "its woody warm and balsamic aroma"
db2.similarity_search_with_score(query, k=3, fetch_k=1)
```

This section demonstrates a similarity search using the Chroma vector store with a specified query. The result includes the most similar items along with their similarity scores.

### 9. Initializing Replicate Language Model
```python
from langchain.

***To run the code***, follow these steps:

Download the CSV file containing the data you want to use for conversational search and question answering. 

Set the Replicate API token:
Set the Replicate API token as an environment variable. You can obtain your API token from your Replicate account settings.

(REPLICATE_API_TOKEN=r8_SG8aoSUQkMOP91vcxYbd00YLv64rHMg1En6Qy)

Run the code:
Run the code using a Python interpreter. The code will perform conversational searches and answer questions based on the provided data.

This code will continuously prompt the user for input and provide answers based on the user's queries. The qa function is responsible for generating answers using the Replicate language model and the Chroma vector store.

**Results-**
(i) Result 1:
---------------------------------------------------------------------------------
Welcome to the BigBasket QueryAgent. You are now ready to start looking for the products you actually need
---------------------------------------------------------------------------------
Prompt: suggest some spicy eatables
Answer:  Based on the given options, I would suggest going for the "Kashmiri Chilli Powder/Menasina Pudi" by Orika. This product has a rating of 4.4 out of 5 stars and is priced at Rs. 42, which is quite reasonable considering its quality and effectiveness.

The perks of using this product include:

1. Adds a vibrant red color to dishes: Unlike regular red chili powders, Kashmiri chili powder imparts a deep red color to food, making it visually
---------------------------------------------------------------------------

(ii) Result 2:
---------------------------------------------------------------------------------
Welcome to the BigBasket QueryAgent. You are now ready to start looking for the products you actually need
---------------------------------------------------------------------------------
Prompt: my hairs are getting thinner, suggest a good shampoo
Answer:  Based on your query, I would suggest going for the "Sunsilk Long & Healthy Growth Shampoo" among the given options. This shampoo has been co-created with Dr. Francesca Fusco, a scalp expert from New York, and comes with an exclusive formula that deeply nourishes the hair from root to tip, keeping it healthy as it grows. The shampoo also contains biotin, which helps reduce hair breakage, making it ideal for people struggling with thinning hair.

(iii) Result 3:
---------------------------------------------------------------------------------
Welcome to the BigBasket QueryAgent. You are now ready to start looking for the products you actually need
---------------------------------------------------------------------------------
Prompt: Suggest a non-fictional novel
Answer:  Based on your input, I would suggest the following non-fictional novel:

"The Immortal Life of Henrietta Lacks" by Rebecca Skloot

This book tells the story of Henrietta Lacks, a poor black tobacco farmer whose cancer cells were taken without her knowledge or consent and led to numerous scientific breakthroughs. The book raises important questions about medical ethics, race, and the intersection of science and society. It has received critical acclaim and won several awards, including the National Book Award for Nonfiction.

(iv) Result 4:
---------------------------------------------------------------------------------
Welcome to the BigBasket QueryAgent. You are now ready to start looking for the products you actually need
---------------------------------------------------------------------------------
Prompt: I am pursuing vegan diet, Suggest snacks
Answer:  Based on your preference for vegan snacks, I would recommend the "Roasted High Protein Mixture" by GoodDiet. It's a tasty and healthy snack option that combines the goodness of grams, lentils, soya nuts, and roasted chole, making it a great source of protein, antioxidants, and fiber. With 100% oil-free and guilt-free, this snack serves well to keep you full and energized throughout the day. Plus, it's free from preservatives and is

(v) Result 5:
---------------------------------------------------------------------------------
Welcome to the BigBasket QueryAgent. You are now ready to start looking for the products you actually need
---------------------------------------------------------------------------------
Prompt: q
*** Tank you for visiting Big Basket ***


