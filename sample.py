import streamlit as st
import pandas as pd

df = pd.DataFrame({
    "Speed": [60, 80, 100],
    "RPM": [2000, 2500, 3000],
    "Style": ["Eco", "Normal", "Aggressive"]
})

st.dataframe(df)
st.write("This is a sample Streamlit app to display a DataFrame.")