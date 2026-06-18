import pandas as pd
import os
from config import * 

def load_data(df_name, data_dir="data"):
    """
    Load a csv file anf return a DataFrame.
    """
    file_name = None
    for fn, name in TABLE_FILES.items():
        if name == df_name:
            file_name = fn
            break
    
    if file_name is None:
        raise ValueError(f"Table name '{df_name}' not found in config.TABLE_FILES.")
    
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found at {file_path}")
        
    print(f"Loading {file_name}...")
    
    return pd.read_csv(file_path, low_memory=False)