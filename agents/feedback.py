def feedback_agent_ice(input_data, model_type_choosen):
    from langchain.llms import Together
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
    driving_style = predict_input_ice(input_data)
    vehicle_type = model_type_choosen  # Fixed variable name

    # Define prompt
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

    # LLM setup
    llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",  # ✅ Fast and free
    temperature=0.7,
    max_tokens=300,
    together_api_key=os.getenv("TOGETHER_API_KEY")
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain
    llm_result = chain.run(vehicle_type=vehicle_type, driving_style=driving_style)

    return llm_result
