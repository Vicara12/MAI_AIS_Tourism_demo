import streamlit as st
import pandas as pd
from dataloader import readTourismData

# Sample list of coordinates
locations = [
    {"lat": 41.3851, "lon": 2.1734},  # Barcelona
    {"lat": 42.2660, "lon": 2.9616},  # Figueres
    {"lat": 41.1189, "lon": 1.2445},  # Tarragona
]

all_data = readTourismData()

# Convert to DataFrame
coords = pd.DataFrame(all_data, columns=['lat', 'lon'])


st.title("Map of Selected Locations")
st.map(coords)
