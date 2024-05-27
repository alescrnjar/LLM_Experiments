In this repository, I explore several prompt engineering techniques made possible by the library LangChain. Each folder contains a separate case, which I describe in the following:

# SQL Table Agent

Transform a question about a dataset into a SQL query, and apply the query to the dataset.
The custom dataset "test_dataset.csv" contains information about age, marriage status and occupation, for 8 datapoints.

# Agents

Use LLM Agents for question answering by using tools such as math calculations, access to private information, consultation of webpages, et cetera.
Robustness to hallucinations is also tested.

# RAG

Perform a simple Retrieval Augmented Generation (RAG) to ask a question relative to the content of an example document.

# Required Libraries 

* langchain 0.1.11
* openai 1.11.1
* langchainhub-0.1.15