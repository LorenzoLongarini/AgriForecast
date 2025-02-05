from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def prepare_data(file_path, test_size=0.2, keep_dateandtime=False, scaled=True, minmax=True):


    data = pd.read_csv(file_path)
    columns_to_remove = ['hour', 'hout', 'etpxkc_tot']
    data = data.drop(columns=columns_to_remove, errors='ignore')

    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    dateandtime_col = None
    if keep_dateandtime and 'dateandtime' in data.columns:
        dateandtime_col = data['dateandtime']

    numeric_data = numeric_data.dropna()
    if dateandtime_col is not None:
        dateandtime_col = dateandtime_col.loc[numeric_data.index]

    split_idx = int(len(numeric_data) * (1 - test_size))

    train_data = numeric_data.iloc[:split_idx].copy()
    test_data = numeric_data.iloc[split_idx:].copy()


    if scaled:
        scaler = StandardScaler() if not minmax else MinMaxScaler()
        train_data = pd.DataFrame(
            scaler.fit_transform(train_data),
            columns=train_data.columns
            # index=train_data.index
        )
        test_data = pd.DataFrame(
            scaler.transform(test_data),
            columns=test_data.columns
            # index=test_data.index
        )

    if dateandtime_col is not None:
        train_data = train_data.assign(dateandtime=dateandtime_col.iloc[:split_idx].values)
        test_data = test_data.assign(dateandtime=dateandtime_col.iloc[split_idx:].values)
    
    return train_data, test_data

if __name__ == "__main__":
    file_path = "../../assets/datimerged/A84041A9018859AC_merged.csv"
    
    keep_dateandtime = True 
    
    train_data, test_data = prepare_data(file_path, keep_dateandtime=keep_dateandtime, scaled=True, minmax=True)
    print(train_data.head(), len(test_data))    



