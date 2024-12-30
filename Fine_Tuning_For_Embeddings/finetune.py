import json
from llama_index.core.readers import SimpleDirectoryReader #pip install llama_index==0.10.19 llama_index_core==0.10.19
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.legacy.finetuning import generate_qa_embedding_pairs, SentenceTransformersFinetuneEngine #pip install llama_index==0.10.19 llama_index_core==0.10.19
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core import Document
import os

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd

def load_corpus_txt(files, verbose=False):
    """
    Load and process text files into nodes.

    Args:
        files (list of str): List of paths to .txt files.
        verbose (bool): Whether to print verbose output.

    Returns:
        list: Parsed nodes from the text files.
    """
    if verbose:
        print(f"Loading files {files}")

    # Read text files and create Document objects
    docs = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        docs.append(Document(text=content))  # Create Document instances

    if verbose:
        print(f"Loaded {len(docs)} docs")

    # Split documents into nodes
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


if __name__=='__main__':

    TRAIN_FILES=[
        'aggregated_f7KSfjv4Oq0.txt', # What Happens If We Throw an Elephant From a Skyscraper? Life & Size 1
        'aggregated_QImCld9YubE.txt', # Why are you alive - Life, Energy and ATP
        'aggregated_QOCaacO8wus.txt', # What is life? Is death real?
        'aggregated_TYPFenJQciw.txt', # The Most Complex Language in the World
        ]   
    VAL_FILES = [
        'aggregated_MUWUHf-rzks.txt', # How to Make an Elephant Explo d – The Size of Life 2    
    ]

    TRAIN_CORPUS_FPATH = "./data/train_corpus.json"
    VAL_CORPUS_FPATH = "./data/val_corpus.json"

    train_nodes = load_corpus_txt(TRAIN_FILES, verbose=True)
    val_nodes = load_corpus_txt(VAL_FILES, verbose=True)

    key_file='../../LM_Tests/openai_api_key.txt'
    with open(key_file, 'r') as file:
        openai_api_key=file.read()
    os.environ["OPENAI_API_KEY"] = openai_api_key
    from llama_index.llms.openai import OpenAI

    # Select LLM for generating Q&A corpus.
    llm=OpenAI(model="gpt-3.5-turbo")

    train_dataset = generate_qa_embedding_pairs(train_nodes, llm=llm)
    train_dataset.save_json("train_dataset.json")

    val_dataset = generate_qa_embedding_pairs(val_nodes, llm=llm)
    val_dataset.save_json("val_dataset.json")


    def finetune(train_dataset,val_dataset):

        finetune_engine = SentenceTransformersFinetuneEngine(
            train_dataset,
            model_id="BAAI/bge-small-en",
            model_output_path="test_model",
            val_dataset=val_dataset,
        )

        #2. Run the Fine-Tuning Process
        #Once the fine-tuning engine is set up, we proceed to run the fine-tuning process. This involves iterating through the training data for a specified number of epochs and updating the model’s weights to better capture the nuances of the specific dataset.

        # Run the fine-tuning process
        finetune_engine.finetune()

        #3. Retrieve the Fine-Tuned Model
        #After completing the fine-tuning process, we can retrieve the fine-tuned model. This model can then be used for embedding generation, tailored to the specific characteristics of your training data.

        # Retrieve the fine-tuned model
        embed_model = finetune_engine.get_finetuned_model()

        # Display the fine-tuned model details
        print(f"{embed_model=}")

    finetune(train_dataset,val_dataset)


