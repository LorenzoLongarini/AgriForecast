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

from sklearn.preprocessing import MinMaxScaler


def trainLSTM():
    input_dim = len(pd.read_excel('./assets/MisureAgugliano.xlsx').columns) - 1
    output_dim = input_dim 

    model = LSTMModel(input_dim=input_dim, output_dim=output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(WeatherDataset(train=True, timesteps=50, ), batch_size=32, shuffle=True)

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

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

  
    torch.save(model.state_dict(), './models/lstm_weather_model.pth')


def testLSTM():
        # Imposta il dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determina input e output dimension
    input_dim = len(pd.read_excel('./assets/MisureAgugliano.xlsx').columns) - 1
    output_dim = input_dim 

    # Inizializza il modello e carica i pesi
    model = LSTMModel(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load('./models/lstm_weather_model.pth'))
    test_loader = DataLoader(WeatherDataset(train=False, timesteps=5), batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())

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