In this repository, I explore several prompt engineering techniques made possible by the library LangChain. Each folder contains a separate case, which I describe in the following:

# SQL Table Agent

Transform a question about a dataset into a SQL query, and apply the query to the dataset.
The custom dataset "test_dataset.csv" contains information about age, marriage status and occupation, for 8 datapoints.

# Agents

Use LLM Agents for question answering by using tools such as math calculations, access to private information, consultation of webpages, et cetera.
Robustness to hallucinations is also tested.

# RAG

Perform a simple Retrieval Augmented Generation (RAG) to ask a question relative to the content of an example document. A minimal script is provided, with the following workflow:
* Divide the input PDF into chunks of size 1000, with no overlap.
* Generate a retriever object with the FAISS vector store and with cosine similarity as the distance strategy.
* Extract the top scoring k=4 chunks, i.e. the most relevant for answering the user's question.
* Use these chunks to construct the final chain for answering the question.

# Required Libraries 

* langchain 0.1.11
* openai 1.11.1
* langchainhub-0.1.15