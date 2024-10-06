import os
import langchain
import langchain.text_splitter 
import langchain_openai
import langchain_community
import langchain_community.vectorstores

from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import ChatOpenAI
from langchain.schema import Document

############################################ RAG Utility Functions

def get_splits(objects,chunk_size=1000, chunk_overlap=0): 
    all_splits=[]
    for obj in objects:
        loader = PyPDFLoader(obj)
        pages_documents = loader.load_and_split() 
        text_splitter = langchain.text_splitter.CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_splits = text_splitter.split_documents(pages_documents)
        print(f"Number of text splits: {len(text_splits)}")
        for spl in text_splits:
            spl.page_content=spl.page_content.replace('\n',' ')
            all_splits.append(spl)
    return all_splits

def get_retriever_for_text_splits(all_splits, k_to_retrieve=4, vectorstore='FAISS', distance_strategy='COSINE'): # ['COSINE','EUCLIDEAN_DISTANCE','MAX_INNER_PRODUCT','DOT_PRODUCT','JACCARD']: #https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.utils.DistanceStrategy.html#langchain_community.vectorstores.utils.DistanceStrategy
    embeddings = langchain_openai.OpenAIEmbeddings()
    if vectorstore=='FAISS':
        db = langchain_community.vectorstores.FAISS.from_documents(all_splits, embeddings, distance_strategy=distance_strategy)
    retriever = db.as_retriever(search_kwargs={'k': k_to_retrieve})
    return retriever

############################################# Graph Plotting Utility Functions

import matplotlib.pyplot as plt
import networkx as nx

def custom_graph_to_networkx(G):
    nx_graph = nx.DiGraph()        # Create an empty directed graph
    for node_id, node_obj in G.nodes.items():
        nx_graph.add_node(node_id) 
    for edge in G.edges:
        nx_graph.add_edge(edge.source, edge.target) 
    return nx_graph

def plot_graph(G,pngname='Graph.png'):
    pos = nx.spring_layout(G)  
    fig=plt.figure(figsize=(12, 8)) 
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    fig.savefig(pngname,dpi=200)
    plt.show()

############################################# LangGraph 

def set_llm():
    #return ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta"))   
    return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)

from typing_extensions import TypedDict
from typing import List
class GraphState(TypedDict):
    question : str
    generation : str
    documents : List[str]

def retrieve(state):
    print("---RETRIEVE FROM PDFS---")
    pdfs=['/home/acrnjar/Downloads/papers/Sasse2023.pdf']
    splits=get_splits(pdfs,chunk_size=1000, chunk_overlap=0)
    retriever=get_retriever_for_text_splits(splits, k_to_retrieve=10, vectorstore='FAISS', distance_strategy='COSINE')
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}
#
def generate(state):
    print("---GENERATE---")
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""", input_variables=["question", "document"])
    llm=set_llm()   
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}
#
def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
    )
    llm=set_llm()
    retrieval_grader = prompt | llm | JsonOutputParser()
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade.lower() == "yes": # Document relevant
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else: # Document not relevant
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


#########################################################
def route_question(state): 
    """
    This function can be adapted so that the initial question can lead to different nodes.
    """
    return "vectorstore"

if __name__=='__main__':

    key_file='../openai_api_key.txt'
    with open(key_file, 'r') as file:
        openai_api_key=file.read()
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # key_file='../huggingfacehub_api_token.txt'
    # with open(key_file, 'r') as file:
    #     openai_api_key=file.read()
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = openai_api_key

    # Define the Workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("NODE_retrieve", retrieve) # retrieve
    workflow.add_node("NODE_grade_documents", grade_documents) # grade documents
    workflow.add_node("NODE_generate", generate) # generatae
    workflow.set_conditional_entry_point(route_question,{"vectorstore": "NODE_retrieve"})
    workflow.add_edge("NODE_retrieve", "NODE_grade_documents")
    workflow.add_edge("NODE_grade_documents","NODE_generate")
    workflow.add_edge("NODE_generate",END)
    ##workflow.add_conditional_edges("NODE_grade_documents",decide_to_generate,{"NODE_websearch": "websearch","NODE_generate": "generate"})
    ##workflow.add_conditional_edges("NODE_generate",grade_generation_v_documents_and_question,{"not supported": "NODE_generate","useful": END, "not useful": "NODE_websearch"})
    app = workflow.compile() # Compile the workflow

    #Test the Workflow
    inputs = {"question": 
            "What is a limitation of the current models?"
            }
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")
    print(value["generation"])

    # Visualize the Agent / Graph
    plot_graph(custom_graph_to_networkx(app.get_graph()))