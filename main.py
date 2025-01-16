import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('./src/dataloader')
sys.path.append('./src/models')
from weather_station import WeatherDataset
from lstm import LSTMModel
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def trainLSTM():
    input_dim = len(pd.read_excel('./assets/MisureAgugliano.xlsx').columns) - 1
    output_dim = input_dim 

    model = LSTMModel(input_dim=input_dim, output_dim=output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(WeatherDataset(train=True, timesteps=15), batch_size=32, shuffle=True, drop_last=False)

    epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}]')
            #   , Loss: {loss.item():.4f}')

  
    torch.save(model, './models/lstm_weather_model.pth')

w_dataset = WeatherDataset(train=False, timesteps=15)

def testLSTM():
        # Imposta il dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determina input e output dimension
    input_dim = len(pd.read_excel('./assets/MisureAgugliano.xlsx').columns) - 1
    output_dim = input_dim 

    # Inizializza il modello e carica i pesi
    # model = LSTMModel(input_dim=input_dim, output_dim=output_dim).to(device)
    model = torch.load('./models/lstm_weather_model.pth', weights_only=False)
    test_loader = DataLoader(w_dataset, batch_size=1, shuffle=False, drop_last=False)
    scaler = w_dataset.scaler
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        predictions = []
        actuals = []
        loss_list = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions.append(scaler.inverse_transform(outputs.numpy()))
            actuals.append(scaler.inverse_transform(targets.numpy()))
            loss_list.append(loss.item())
        print(np.mean(loss_list))

    # Salva le predizioni e i target per la valutazione in un file
    results = {
        'predictions': predictions,
        'actuals': actuals
    }
    torch.save(results, './models/test_results.pth')
    print('Risultati di test salvati per la valutazione.')

def main():
    trainLSTM()
    testLSTM()

if __name__ == "__main__":
    main()