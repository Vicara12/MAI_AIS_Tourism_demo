import pandas as pd

def setFromColValues(df: pd.DataFrame, column: str) -> set:
  return set(df[column])