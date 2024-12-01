import pandas as pd
import torch
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class WeatherDataset(Dataset):
    def __init__(self,  train_len = .8, train = True, path = './/assets//MisureAgugliano.xlsx', timesteps=20, window_size = 1):

        self.scaler = StandardScaler()
        self.window_size = window_size
        self.timesteps = timesteps
        self.data = pd.read_excel(path)        
        self.data = self.data.drop(columns=['day'])
        self.data = self.data.replace(',', '.', regex=True).astype(float)

        train_len = math.ceil(len(self.data) * train_len)
        train_data = self.data[:train_len]
        self.scaler.fit(train_data)


        if train:
            self.data = train_data
            self.data.iloc[:, :] = self.scaler.transform(self.data)
        else:
            self.data = self.data[train_len:]
            self.data.iloc[:, :] = self.scaler.transform(self.data)


        self.x, self.y = self.create_sequences()
    
    def create_sequences(self):
        x = []
        y = []
        for i in range(len(self.data)-self.timesteps):
            x.append(self.data.iloc[i:i+self.timesteps- self.window_size].values)
            y.append(self.data.iloc[i+self.timesteps].values)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

