In this repository, I explore several prompt engineering techniques made possible by the library LangChain. Each directory contains a separate case, which I describe in the following:

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

# Knowledge Graph Extraction + Q & A

Extract a knowledge graph from an example text, and perform question answering based on the extracted graph. The script makes use of LLMGraphTransformer from langchain_experimental (version: 0.0.62). The knowledge graph is plotted with libraries networkx and matplotlib.

# LangGraph

LangGraph allows for the engineering of the prompt chain. Operations such as document retrieval, web retrieval, generation, evaluation, et cetera are defined as nodes on a graph that can be directly designed by the user.  In the dedicated directory, I provide a simple template that can be adjusted for custom cases.

# Fine Tune Embeddings
I use llama-index for fine tuning of an embedding model, to be used in a langchain RAG pipeline for document retrieval, according to the dedicated llama-index tutorial. I follow these steps:
* generate question-answer pairs from the training corpus of text.
* finetune the model
* use the new embeddings in the langchain script.

# GraphRAG

GraphRAG leverages on the construction of a knowledge graph from the text corpus and uses community summaries to answer the user query. This allows for a better performance on questions that benefits from an understanding of the overall text. GraphRAG is divided into two stages:
* Indexing: the actual build up of the communities. It needs to be done only once, at it is typically an expensive step.
* Querying: leverages the communities for addressing the user question. Querying can be done in  either global mode or local mode.
Here, I use microsoft graphrag package to perform both stages: I take care of indexing with a pre made script, then perform a global query in a python script.

# Required Libraries 

* langchain 0.1.11
* openai 1.11.1
* langchainhub-0.1.15
* langchain-community: 0.2.9
* langchain-experimental: 0.0.62
* langgraph: 0.1.8
* (Finetuning) llama-index: 0.10.19
* (Finetuning) llama-index.core: 0.10.19
* (Finetuning) sentence_transformers: 2.2.2