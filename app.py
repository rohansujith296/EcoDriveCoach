import streamlit as st
import pandas as pd
from agents.feedback import feedback_agent_ice
from agents.feedback2 import feedback_agent_ev
import os 
from dotenv import load_dotenv
load_dotenv()


# --- EV Model Input Fields ---
ev_fields = [
    "avg_speed_kmph", "top_speed_kmph", "acceleration_aggressiveness", "braking_hard_events",
    "idle_time_min", "terrain_type", "ambient_temperature_C", "ac_usage_percent",
    "regen_braking_enabled", "eco_mode_enabled", "battery_usage_kWh"
]

ev_types = {
    "avg_speed_kmph": float,
    "top_speed_kmph": float,
    "acceleration_aggressiveness": int,
    "braking_hard_events": int,
    "idle_time_min": float,
    "terrain_type": str,
    "ambient_temperature_C": float,
    "ac_usage_percent": int,
    "regen_braking_enabled": int,
    "eco_mode_enabled": int,
    "battery_usage_kWh": float
}

# --- ICE Model Input Fields ---
ice_fields = [
    "rpm", "speed", "gear_position", "acceleration", "braking", "throttle_position",
    "idle_time", "engine_load", "fuel_rate", "coolant_temp"
]

ice_types = {
    "rpm": float,
    "speed": float,
    "gear_position": int,
    "acceleration": float,
    "braking": float,
    "throttle_position": float,
    "idle_time": int,
    "engine_load": float,
    "fuel_rate": float,
    "coolant_temp": float
}

st.title("EcoDriveCoach")

# ---- EV Input Form ----
def input_ev_fields():
    ev_input = {}
    for field in ev_fields:
        if field == "terrain_type":
            ev_input[field] = st.selectbox("Terrain Type", ["hilly", "flat", "mixed"])
        elif field in ["regen_braking_enabled", "eco_mode_enabled"]:
            ev_input[field] = st.selectbox(field.replace("_", " ").capitalize(), [0, 1])
        else:
            ev_input[field] = st.number_input(field.replace("_", " ").capitalize(), value=0.0 if ev_types[field] == float else 0)
    return ev_input

# ---- ICE Input Form ----
def input_ice_fields():
    ice_input = {}
    for field in ice_fields:
        ice_input[field] = st.number_input(field.replace("_", " ").capitalize(), value=0.0 if ice_types[field] == float else 0)
    return ice_input

# ---- Vehicle Type ----
model_type = st.radio("Select Vehicle Type", ("EV", "ICE"))

input_data = None
feedback = None

if model_type == "EV":
    st.header("EV Model Input")
    input_data = input_ev_fields()
elif model_type == "ICE":
    st.header("ICE Model Input")
    input_data = input_ice_fields()

# ---- Submit Button ----
if st.button("Submit"):
    st.write("Submitting your data...")
    st.success("Successfully submitted. Waiting for response...")

    if model_type == "EV":
        feedback = feedback_agent_ev(input_data, model_type)
    elif model_type == "ICE":
        feedback = feedback_agent_ice(input_data, model_type)

    st.subheader("🧠 EcoDrive Feedback:")
    st.write(feedback)
