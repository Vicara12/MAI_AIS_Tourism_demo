import pandas as pd
import os

def readTourismData(folder_path: str = "data") -> pd.DataFrame:
    merged_df = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            municipality = filename.split("_")[1]
            file_path = os.path.join(folder_path, filename)
            temp_df = pd.read_csv(file_path)
            temp_df['municipality'] = municipality
            merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
    return merged_df