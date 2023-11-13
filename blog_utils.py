from Bio import Entrez
import streamlit as st
# import metapub
from datetime import datetime
import os
import openai
import configparser
from pathlib import Path
from langchain.chains import StuffDocumentsChain
from langchain.document_loaders import TextLoader
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from bs4 import BeautifulSoup
import requests
import random
from functools import wraps 
# from dotenv import load_dotenv
# load_dotenv()
# openai.api_key=os.getenv("OPENAI_API_KEY")
# PUBMED_API_KEY=os.getenv("PUBMED_API_KEY")
# Entrez.email = os.getenv("email")
import logging

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

# cfg_reader = configparser.ConfigParser()
# fpath = Path.cwd() / Path('config.ini')
# cfg_reader.read(str(fpath))
# Entrez.api_key = cfg_reader.get('API_KEYS','PUBMED_API_KEY')
# openai.api_key = cfg_reader.get('API_KEYS','OPENAI_API_KEY')


def is_valid_api_key():
    #openai.api_key = api_key
    try:
        models = openai.Model.list()
        return True
    except openai.error.AuthenticationError as e:
        return False
#try:
#    Entrez.email = 'drsheraz@xevensolutions.com'
##    Entrez.api_key = os.getenv('PUBMED_API_KEY')
#    openai.api_key = os.getenv('OPENAI_API_KEY')
#openai.api_key = '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'


def openai_api_key_required(func):
    """
    Decorator that requires an OpenAI API key to be set before executing the decorated function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not openai.api_key:
            raise ValueError("OpenAI API key must be set using openai.api_key before calling this function.")
        
        try:
            models = openai.Model.list()
            return func(*args, **kwargs)
        except openai.error.AuthenticationError as e:
            raise ValueError("Invalid OpenAI API key. Please Replace Your Key")
    return wrapper


def get_response_from_openai(prompt, ai_engine):
    response = openai.Completion.create(
                    engine=ai_engine,
                    prompt=prompt,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
    return response

@openai_api_key_required
def get_blogpost_from_gpt3_pubmed(search_term,title_list, ai_engine):
    #if not (is_valid_api_key()):
    #    return 'Invlaid API Key or OpenAI Authentication Error'
    #preamble = f"Generate a blog post of 500 words on {search_term}, synthesizing information from the following Article List and incorporating in-text citations and a references section at the end."
    preamble = f"Write and Article of 500 words on {search_term}, synthesizing information from the following Article List and incorporating in-text citations and a references section at the end. Also generate the title of Article."
        # conversion of title list to string for prompt generation
    title_string = ''
    for i, string in enumerate(title_list):
        title_string += f'{i+1}. {string} '
        #footer = "Only return the appropriate label from 'Risky', 'Partially-Risky' or 'Safe'  and do not write any additional text."
    try:
        open_ai_request = preamble+ " Article List: " + title_string #+ ". " + footer
        response = get_response_from_openai(open_ai_request, ai_engine)
        print('Response received from openAI ...')
        return response.choices[0].text
    except Exception as e:
        print(f"Error {str(e)} from OpenAI")

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

#urls against pubmed ids
def generate_urls_from_pubmed_ids(id_list):
    base_url="https://pubmed.ncbi.nlm.nih.gov/"
    url_list = [base_url + id + "/" for id in id_list]
    return url_list

#function to summarize abstract
@openai_api_key_required
def generate_abstract_summary(all_texts):
    generated_list=[]
    for i in range(len(all_texts)):
        try:
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
            

        #Handle API error here, e.g. retry or log    
        except openai.error.APIError as e:
            logger.critical(f"OpenAI API returned an API Error")
            raise openai.error.APIError('OpenAI API returned an API Error')
        #Handle connection error here
        except openai.error.APIConnectionError as e:
            logger.critical(f"Failed to connect to OpenAI API")
            raise openai.error.APIConnectionError('Failed to connect to OpenAI API')
        #Handle rate limit error (we recommend using exponential backoff)
        except openai.error.RateLimitError as e:
            logger.critical(f"OpenAI API request exceeded rate limit")
            raise openai.error.RateLimitError('OpenAI API request exceeded rate limit')
        except Exception as e:
            logger.critical(f'Error : {str(e)}')
            raise Exception("AI is not available")
        except Exception as ex:
            logger.critical(f'Generate Summary Unsuccessful: {str(ex)}')
            raise Exception(f"error : OPENAPI KEY INVALID {str(ex)}")
    return generated_list


#scrapping function for full text against urls
def article_full_text_scrapping(urls):
    generated_list=[]
    for i in range(len(urls)):
        url=urls[i]
        response=requests.get(url)
        htmlContent=response.content
        soup = BeautifulSoup(htmlContent, 'html.parser')
        link = soup.find('a', class_='link-item')
        target_link = link.get('href')
        if target_link=='#':
            generated_list.append("Full text of this article is not available")
        else:
            generated_list.append(target_link)
    return generated_list





#function to generate abstract summary using langchain
def generate_abstract_summary_langchain(doc):
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    loader = TextLoader(doc)
    docs=loader.load()
    response=stuff_chain.run(docs)
    return response










#test functions
def get_PMIDs_for_term_free(s_term, pub_date_range='2018/01/01:2021/12/31'):
    n = 10
    database = "pubmed"
    search_results = Entrez.esearch(db=database, term=s_term, retmax=n, sort="relevance", mindate=pub_date_range)
    record = Entrez.read(search_results)
    ids = record['IdList']

    # Initialize a list to store PMIDs of freely available articles
    freely_available_ids = []

    # Check if articles are freely available
    for pmid in ids:
        # Use Entrez.efetch to retrieve the article summary
        handle = Entrez.efetch(db=database, id=pmid, retmode="xml")
        record = Entrez.read(handle)

        # Check if the article is freely available
        if "PubStatus" in record[0].get("PubStatus", []):
            pub_status = record[0]["PubStatus"]
            if "aheadofprint" in pub_status or "epublish" in pub_status:
                # This article is freely available
                freely_available_ids.append(pmid)

    return freely_available_ids

def esummary(ids):
    response=[]
    try:
        for article_uid in ids:
            # Use the esummary function to fetch a summary of each article
            summary_result = Entrez.esummary(db="pmc", id=article_uid)
            summary_record = Entrez.read(summary_result)

            # Print the summary information for each article
            print(summary_record)

            # Close the summary result
            summary_result.close()
    except Exception as e:
        print("An error occurred:", e)

def get_PMIDs_for_term_test(s_term, pub_date_range='2023/09/01:2023/11/1'):
    n = 10
    database = "pubmed"
    search_results = Entrez.esearch(db=database, term=s_term, retmax=n, sort="relevance", mindate=pub_date_range)
    record = Entrez.read(search_results)
    ids = record['IdList']
    return ids


#-----------pmc---------------
#get pmc articles ids
def get_PMCs_for_term(s_term, n=10,pub_date_range='2018/01/01:2021/12/31',pubtype="Review,Systematic Review,Clinical Trial", Lang='eng'):
    database = "pmc"
    search_results = Entrez.esearch(db=database, term=s_term, retmax=n, sort="relevance", mindate= pub_date_range)
    record = Entrez.read(search_results)
    ids = record['IdList']
    return ids
#get article title from pumc
def get_title_list_from_pmc_id_list(pmc_id_list):
    assert isinstance(pmc_id_list, list), "Input must be a list"
    title_list = []
    with Entrez.efetch(db="pmc", id=pmc_id_list, retmode="xml") as handle:
        records = Entrez.read(handle)
    for record in records:
        if "title" in record and "title-group" in record["title"]:
            title_list.append(record["title"]["title-group"][0]["article-title"])
    return title_list

@openai_api_key_required    
def get_blogpost_from_chatgpt_pubmed(search_term:str, title_list:list) -> str:
    #if not (is_valid_api_key()):
    #    return 'Invlaid API Key or OpenAI Authentication Error'
    
    preamble=f"Write an Article of approximately 500 words on '{search_term}' with in-text citations. Synthesize information from the following Article List. Inclusion of in-text 'citations' of provided Article List is mandatory. Must Include a 'References' section at the end."
    
    #preamble = f"Write an Article of 500 words on {search_term} with in-text citations. Synthesize information from the following Article List. Include in-text citations of provided Article List and a references section at the end."
        # conversion of title list to string for prompt generation
    title_string = ''
    for i, string in enumerate(title_list):
        title_string += f'{i+1}. {string} \n '
        prompt = preamble+ " Article List: " + title_string
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant that writes blogposts on given topics."},
                {"role": "user", "content":prompt }
            ],
        temperature = 0.2
        )
    except Exception as e:
        print(f'Error : {str(e)}')
    blogpost = response['choices'][0]['message']['content']
    if blogpost is not None:
        return blogpost
    else:
        return None

@openai_api_key_required
def get_blogpost_from_chatgpt_prompt (mprompt):
    assert isinstance(mprompt, str)
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant that writes blogposts on given topics."},
                {"role": "user", "content":mprompt }
            ],
        temperature = 0.2
        )
    except Exception as e:
        print(f'Error : {str(e)}')
    blogpost = response['choices'][0]['message']['content']
    if blogpost is not None:
        return blogpost
    else:
        return None

@openai_api_key_required    
def get_blogpost_from_chatgpt_topic (search_term):
    #preamble= f"Write an Article of approximately 500 words on '{search_term}' with in-text citations. Inclusion of in-text 'citations' of Latest Research is mandatory. Must Include a 'References' section at the end."
    prompt = f"Write an Article of approximately 500 words on '{search_term}' with in-text citations. Inclusion of in-text 'citations' of Latest Research is mandatory. Must Include a 'References' section at the end. Also give a catchy Title."
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant that writes blogposts on given topics."},
                {"role": "user", "content":prompt }
            ],
        temperature = 0.2
        )
    except Exception as e:
        print(f'Error : {str(e)}')
    blogpost = response['choices'][0]['message']['content']
    if blogpost is not None:
        return blogpost
    else:
        return None

#function to get url against Pubmed IDs
def generate_urls_from_ids(id_list, base_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"):
    url_list = [base_url + id + "/" for id in id_list]
    return url_list
#test
def search_pubmed(query, retmax=20):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    record = Entrez.read(handle)

    for pmid in record["IdList"]:
        summary = Entrez.esummary(db="pubmed", id=pmid)
        article_info = Entrez.read(summary)[0]
        title = article_info["Title"]
        is_open_access = "OA" in article_info.get("FullJournalName", "")

        if is_open_access:
            print(f"Title: {title}\nStatus: Open Access\n")
        else:
            print(f"Title: {title}\nStatus: Not Open Access\n")


