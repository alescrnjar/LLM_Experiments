import os
import asyncio

import pandas as pd
import tiktoken

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

def global_graphrag(
                    INPUT_DIR, 
                    question="What is the major conflict in this story and who are the protagonist and antagonist?",
                    llm_model='gpt-4o-mini',
                    COMMUNITY_LEVEL = 2, # community level in the Leiden community hierarchy from which we will load the community reports: higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
                    verbose=True,
                    ):  

    llm = ChatOpenAI(api_key=api_key, model=llm_model, api_type=OpenaiApiType.OpenAI, max_retries=20)
    token_encoder = tiktoken.get_encoding("cl100k_base")

    COMMUNITY_REPORT_TABLE = "create_final_community_reports" #Load community reports as context for global search. #Load all community reports in the create_final_community_reports table from the ire-indexing engine, to be used as context data for global search.
    ENTITY_EMBEDDING_TABLE = "create_final_entities" #Load entities from the create_final_nodes and create_final_entities tables from the ire-indexing engine, to be used for calculating community weights for context ranking. 
    ENTITY_TABLE = "create_final_nodes" 

    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    if verbose: print(f"Report records: {len(report_df)}")
    if verbose: print(report_df.head())

    #Build global context based on community reports
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    #Perform global search
    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    async def get_result(query=question):
        #result = await search_engine.asearch("What is the major conflict in this story and who are the protagonist and antagonist?")
        result = await search_engine.asearch(query)
        #print(result.context_text)
        # inspect the data used to build the context for the LLM responses
        #print(result.context_data["reports"])
        # inspect number of LLM calls and tokens
        #print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
        return result

    result=asyncio.run(get_result(query=question))
    return result


if __name__=='__main__':

    """
    https://microsoft.github.io/graphrag/posts/query/notebooks/global_search_nb/
    Global Search for reasoning about holistic questions about the corpus by leveraging the community summaries.
    Local Search for reasoning about specific entities by fanning-out to their neighbors and associated concepts.
    """

    inputdir='./parquet_library_safecopies/'
    chosen_model='gpt-4o-mini'
    question='Which speaker faced challenges regarding regulations and informed consent when putting cells into culture?'

    #####

    key_file='../LM_Tests/openai_api_key.txt'
    with open(key_file, 'r') as file:
        openai_api_key=file.read()
    #os.environ["OPENAI_API_KEY"] = openai_api_key

    os.environ["GRAPHRAG_API_KEY"] = openai_api_key #AC
    api_key = os.environ["GRAPHRAG_API_KEY"]
    os.environ["GRAPHRAG_LLM_MODEL"]=chosen_model
    #llm_model = os.environ["GRAPHRAG_LLM_MODEL"]

    result=global_graphrag(
                    inputdir, 
                    question,
                    chosen_model,
                    COMMUNITY_LEVEL = 2, 
                    verbose=True,
                )

    #print(result.context_text)
    # inspect the data used to build the context for the LLM responses
    print(result.context_data["reports"])
    # inspect number of LLM calls and tokens
    print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
    print(f"{result.response=}")

    print("SCRIPT END.")


