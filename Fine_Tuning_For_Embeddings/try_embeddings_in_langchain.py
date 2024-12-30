from sentence_transformers import SentenceTransformer
import langchain_community
import langchain_community.vectorstores
import langchain_openai
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
import os

from langchain.embeddings import HuggingFaceEmbeddings

class MyEmbeddings:
    def __init__(self, model):
        self.model = SentenceTransformer(model)
    
    def embed_documents(self, texts) :
        embeddings = self.model.encode(texts)  
        return embeddings.tolist()  

    def embed_query(self, query):
        return self.model.encode([query])

    def __call__(self, texts):
        return self.embed_documents(texts)


if __name__=='__main__':

    key_file='../../LM_Tests/openai_api_key.txt'
    with open(key_file, 'r') as file:
        openai_api_key=file.read()
    os.environ["OPENAI_API_KEY"] = openai_api_key

    key_file='../../LM_Tests/huggingfacehub_api_token.txt'
    with open(key_file, 'r') as file:
        api_key=file.read()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

    texts= [
    'aggregated_f7KSfjv4Oq0.txt', # What Happens If We Throw an Elephant From a Skyscraper? Life & Size 1
    'aggregated_QImCld9YubE.txt', # Why are you alive - Life, Energy and ATP
    'aggregated_QOCaacO8wus.txt', # What is life? Is death real?
    'aggregated_TYPFenJQciw.txt', # The Most Complex Language in the World
        ]
    docs=[]
    for text in texts:
        loader = TextLoader(text)
        doc=loader.load()
        docs.append(doc[0])

    query='How is a mouse different from an elephant?'

    for emb_model in ['OpenAI','BAAI/bge-small-en','MyEmbeddings']:
        print("=== === ===",emb_model)

        if emb_model=='OpenAI':
            embeddings = langchain_openai.OpenAIEmbeddings() 
        
        if emb_model=='BAAI/bge-small-en':
           embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        
        if emb_model=='MyEmbeddings':
            embeddings = MyEmbeddings('test_model')

        db = langchain_community.vectorstores.FAISS.from_documents(docs, embeddings, distance_strategy='COSINE')

        retr=db.similarity_search_with_score(query=query,search_kwargs={'k': 4})

        # Visualize the similarity scores of the retrieved documents, according to the used embedding model.
        for ret in retr:
            source=ret[0].metadata['source']
            score=ret[1]
            print(source,score)

    print("SCRIPT END.")




