import pandas as pd
import numpy as np

def prepare_data(file_path, test_size=0.2, keep_dateandtime=False):


    data = pd.read_csv(file_path)
    if keep_dateandtime:
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if 'dateandtime' in data.columns:
            numeric_data['dateandtime'] = data['dateandtime']
    else:
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

    numeric_data = numeric_data.dropna()

    split_idx = int(len(numeric_data) * (1 - test_size))

    train_data = numeric_data.iloc[:split_idx]
    test_data = numeric_data.iloc[split_idx:]
    
    return train_data, test_data

if __name__ == "__main__":
    file_path = "../../assets/datimerged/A84041A9018859AC_merged.csv"
    
    keep_dateandtime = False 
    
    train_data, test_data = prepare_data(file_path, keep_dateandtime=keep_dateandtime)
    print(len(train_data), len(test_data))    



