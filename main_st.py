import re
import os
import json
import urllib.request
import streamlit as st
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()


# Function to fetch papers from arXiv API
def fetch_papers():
    url = 'https://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    root = ET.fromstring(data)

    papers_list = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        paper_info = f"Title: {title}\nSummary: {summary}\n"
        papers_list.append(paper_info)

    return papers_list


# Save a list to a JSON file
def save_list_to_json(list_of_strings, filename):
    with open(filename, 'w') as file:
        json.dump(list_of_strings, file)


# Load a list from a JSON file
def load_list_from_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# Initialize the QA chain and FAISS index
def initialize_qa_chain():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model = "text-embedding-ada-002"
    embeddings_model = OpenAIEmbeddings(model=model, api_key=openai_api_key)

    try:
        papers_list = load_list_from_json("papers_list.json")
    except:
        papers_list = fetch_papers()
        save_list_to_json(papers_list, "papers_list.json")

    try:
        db = FAISS.load_local("faiss_index")
    except:
        db = FAISS.from_texts(papers_list, embeddings_model)
        db.save_local("faiss_index")

    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.5})
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Name of the language model
        temperature=0,  # Parameter that controls the randomness of the generated responses
        api_key=openai_api_key
    )
    prompt_template = "{context}\n\nQuestion: {question}"
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["summaries", "question"],
            ),
        }
    )

    return qa


# Streamlit UI
st.title('Llama 2 QA System')
query = st.text_input("Enter your query:")

if st.button('Get Answer') and query:
    # Initialize QA chain
    qa = initialize_qa_chain()

    # Fetch response
    response = qa(query)

    # Display the response and source papers
    st.subheader("Answer:")
    st.write(response["result"])

    st.subheader("Source Papers:")
    titles = [re.search(r'Title: (.+?)\n', paper.page_content).group(1) for paper in response["source_documents"]]
    for e, title in enumerate(titles):
        st.write(str(e + 1) + "- " + title)
