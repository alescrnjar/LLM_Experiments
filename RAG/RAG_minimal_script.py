import langchain
import langchain.text_splitter 
from langchain_community.document_loaders import PyPDFLoader
import langchain_community.vectorstores
import langchain_openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def get_text_splits_for_pdf_files(pdf_files, chunk_size=1000, chunk_overlap=0):
    all_splits=[]
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages_documents = loader.load_and_split() 
        text_splitter = langchain.text_splitter.CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_splits = text_splitter.split_documents(pages_documents)
        for spl in text_splits:
            all_splits.append(spl)
    return all_splits

def get_retriever_for_pdf_files(all_splits, k_to_retrieve=4, vectorestore='FAISS', distance_strategy='COSINE'): # ['COSINE','EUCLIDEAN_DISTANCE','MAX_INNER_PRODUCT','DOT_PRODUCT','JACCARD']: #https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.utils.DistanceStrategy.html#langchain_community.vectorstores.utils.DistanceStrategy
    embeddings = langchain_openai.OpenAIEmbeddings()
    if vectorestore=='FAISS':
        db = langchain_community.vectorstores.FAISS.from_documents(all_splits, embeddings, distance_strategy=distance_strategy)
    retriever = db.as_retriever(search_kwargs={'k': k_to_retrieve})
    return retriever

def rag_chain(question, retriever, temperature=0):
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=temperature, model="gpt-4")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    answer=chain.invoke(question)
    return answer


if __name__=='__main__':

    import os

    key_file='../openai_api_key.txt'
    with open(key_file, 'r') as file:
        openai_api_key=file.read()
    os.environ["OPENAI_API_KEY"] = openai_api_key

    pdf_files=['~/Downloads/papers/Wei2022.pdf']
    all_splits=get_text_splits_for_pdf_files(pdf_files, chunk_size=1000, chunk_overlap=0)
    retriever=get_retriever_for_pdf_files(all_splits)

    question="Describe the method of Chain of Thought in no more than two sentences."
    answer=rag_chain(question,retriever)
    print(answer)
