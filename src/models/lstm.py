import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm.keras import TqdmCallback
from callbacks import CallbackHandler


def train_lstm_all_features(train_data, test_data, look_back=10, epochs=10, batch_size=32):

    metrics = []
    metrics_callback = CallbackHandler()

    for feature in train_data.columns:

        train_series = train_data[feature].values
        test_series = test_data[feature].values

        # scaler = MinMaxScaler()
        # scaled_train = scaler.fit_transform(train_series)
        # scaled_test = scaler.transform(test_series)

        def create_dataset(data, look_back):
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i:i + look_back])
                y.append(data[i + look_back])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_series, look_back)
        X_test, y_test = create_dataset(test_series, look_back)

        X_train = X_train.reshape(-1, look_back, 1)
        X_test = X_test.reshape(-1, look_back, 1)

        model = Sequential([
            LSTM(50, input_shape=(look_back, 1), return_sequences=True),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        metrics_callback.start()
        model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=1,
            callbacks=[TqdmCallback(verbose=1)])
        metrics_callback.stop()
        train_efficency_metric = metrics_callback.collect(key = 'train')
        
        metrics_callback.start()
        predictions = model.predict(X_test)
        metrics_callback.stop()
        test_efficency_metric = metrics_callback.collect(key = 'test')

        # predictions = scaler.inverse_transform(predictions)  
        # y_test = scaler.inverse_transform(y_test.reshape(-1, 1)) 

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse, **train_efficency_metric, **test_efficency_metric})

    white_list = [ 'MAE', 'RMSE'] + list(train_efficency_metric.keys()) + list(test_efficency_metric.keys())

    metrics_df = pd.DataFrame(metrics)
    overall = {c: metrics_df[c].mean() for c in white_list}

    return metrics_df, overall
