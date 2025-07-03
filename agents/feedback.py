from langchain.llms import HuggingFaceEndpoint
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from model import predict_input_ev, predict_input_ice
import streamlit as st
import pandas as pd




with open("/Users/rohansujith/Desktop/Python/EcoDriveCoach/model.py", "rb") as f:
    model_ev = pickle.load(f)
with open("/Users/rohansujith/Desktop/Python/EcoDriveCoach/model.py", "rb") as k:
    model_ice = pickle.load(k)





res_ev = predict_input_ev(ev_input)
res_ice = predict_input_ice(ice_input)



def get_feedback_chain(res_ev, res_ice):
    """
    Create a feedback chain using HuggingFaceEndpoint and FAISS.
    """
    # Define the model endpoint
    model_endpoint = os.getenv("HUGGINGFACE_ENDPOINT", "https://api-inference.huggingface.co/models/your-model-name")

    # Initialize the LLM with the endpoint
    llm = HuggingFaceEndpoint(
        model=model_endpoint,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2
    )

    # Define the prompt template for feedback
    prompt_template = PromptTemplate(
        input_variables=["input_text"],
        template="Provide feedback on the following text: {input_text}"
    )

    # Create the LLM chain
    feedback_chain = LLMChain(llm=llm, prompt=prompt_template)

    return feedback_chain

