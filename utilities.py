import openai
import streamlit as st
from Bio import Entrez
import logging
from functools import wraps 
import random
from datetime import datetime



openapi_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openapi_key
PUBMED_API_KEY=st.secrets["PUBMED_API_KEY"]
Entrez.email=st.secrets["email"]

# Setup the custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('mamsa_utlities.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# def is_valid_api_key():
#     #openai.api_key = api_key
#     try:
#         models = openai.Model.list()
#         return True
#     except openai.error.AuthenticationError as e:
#         return False

# def openai_api_key_required(func):
#     """
#     Decorator that requires an OpenAI API key to be set before executing the decorated function.
#     """
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         if not openai.api_key:
#             raise ValueError("OpenAI API key must be set using openai.api_key before calling this function.")
        
#         try:
#             models = openai.Model.list()
#             return func(*args, **kwargs)
#         except openai.error.AuthenticationError as e:
#             raise ValueError("Invalid OpenAI API key. Please Replace Your Key")
#     return wrapper



def get_PMIDs_for_term(s_term):
    random_number = random.randint(5, 10)
    database = "pubmed"
    start_date = datetime(2023, 10, 8).strftime("%Y/%m/%d")
    end_date = datetime(2023, 11, 8).strftime("%Y/%m/%d")
    search_query = f'{s_term} AND ("{start_date}"[PDAT] : "{end_date}"[PDAT])'
    search_results = Entrez.esearch(db=database, term=search_query, retmax=random_number, sort="relevance")
    record = Entrez.read(search_results)
    ids = record['IdList']
    return ids

def get_abstract_list_from_pmid_list (pmidlist):
    scan_ids=[]
    assert isinstance(pmidlist, list), "Input must be a list"
    abstract_list = []
    with Entrez.efetch(db="pubmed", id=pmidlist, retmode="xml") as handle:
        records = Entrez.read(handle)
    for i,record in enumerate(records["PubmedArticle"]):
        if record["MedlineCitation"]["Article"].get("Abstract",None) is not None:
            abstract_list.append(str(record["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]).replace('[','').replace(']','').replace("'",""))
            scan_ids.append(pmidlist[i])
    return abstract_list,scan_ids

def get_title_list_from_pmid_list(pmidlist):
    assert isinstance(pmidlist, list), "Input must be a list"
    title_list = []
    with Entrez.efetch(db="pubmed", id=pmidlist, retmode="xml") as handle:
        records = Entrez.read(handle)
    for record in records["PubmedArticle"]:
        if record["MedlineCitation"]["Article"].get("ArticleTitle",None) is not None:
            title_list.append(record["MedlineCitation"]["Article"]["ArticleTitle"])
    return title_list

def generate_abstract_summary(all_texts):
    generated_list=[]
    for i in range(len(all_texts)):
            prompt = f"""You are text summarizer who is expert at performing Extreme TLDR generation for given text. 
            Extreme TLDR is a form of extreme summarization, which performs high source compression, removes stop words and
            summarizes the text whilst retaining meaning. The result is the shortest possible summary that retains all of 
            the original meaning and context of the text.The summary wording should be understand by layman.The summary length should be atmost 200 words. 
            text for Extreme TLDR generation : {all_texts[i]}
            Extreme TLDR : """
           
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are an expert in generating summaries for legal documents."},
                        {"role": "user", "content":prompt }
                    ],
                temperature = 0.2
                )
            refined_text = response['choices'][0]['message']['content']
            generated_list.append(refined_text)
    return generated_list