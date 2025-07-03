def feedback_agent_ev(input_data, model_type_choosen):
    from langchain.llms import HuggingFaceEndpoint
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    import os
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from model import predict_input_ev, predict_input_ice
    import streamlit as st
    import pandas as pd
    from dotenv import load_dotenv
    
    
    
    load_dotenv()


    # Predict driving style
    driving_style = predict_input_ev(input_data)
    vehicle_type = model_type_choosen

    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["vehicle_type", "driving_style"],
        template="""
You are EcoDriveCoach, an intelligent driving assistant designed to give personalized feedback based on driving behavior.
Your job is to analyze the user's driving style and offer constructive suggestions to improve energy or fuel efficiency.

## Context:
Vehicle Type: {{ vehicle_type }}
Driver Style: {{ driving_style }}

## Instructions:
- Use a friendly, encouraging tone.
- Provide a short summary of what the driving style means.
- Then, suggest 2 to 3 specific, practical tips to improve efficiency for the given vehicle type.
- If the driver is already driving efficiently (e.g., "Eco"), offer motivational tips to maintain good habits.

## Response Format:
Driving Style Summary:
Tips to Improve:
- ...
- ...
- ...

Begin:
"""
    )

    # Load LLM
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.4,
        max_new_tokens=150,
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

    # Chain the LLM with the prompt
    chain = LLMChain(llm=llm, prompt=prompt)

    # Get response from LLM
    llm_result = chain.run(vehicle_type=vehicle_type, driving_style=driving_style)

    return llm_result
