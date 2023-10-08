# Standard packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error
import time
from sklearn.metrics import mean_absolute_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import plotly.graph_objects as go
sns.set_style('darkgrid')
import os
os.environ['KMP_WARNINGS'] = 'off'
device = torch.device("cuda")


class Classifier(object):
    def __init__(self, model):
        self.model = model
        self.model = self.model.to(device)
    def train(self, train_data, params):
        '''
            Input: train_data - list of input values (numpy array) and target values
                                (numpy array) of training data
                   model - model to be trained
                   show_progress - if the training process is showed (boolean)

        '''

        total_time = 0
        self.x_train, self.y_train = train_data
        self.x_train = self.x_train.to(device)#Chuyển dữ liệu x_test sang GPU
        self.y_train = self.y_train.to(device)#Chuyển dữ liệu y_test sang GPU
        criterion = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        hist = np.zeros(params.n_epochs)
        self.model.train()
        start_time = time.time()
        for epoch in range(params.n_epochs):
            #start_time = time.time()
            y_train_pred = self.model(self.x_train)
            loss = criterion(y_train_pred, self.y_train)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            #end_time = time.time()
            #elapsed_time = end_time - start_time
            #total_time += elapsed_time
            print(f'Epoch: {epoch+1}/{params.n_epochs}\tMSE loss: {loss.item():.5f} ')
            hist[epoch] = loss.item()  
        end_time = time.time()
        total_time = end_time - start_time 
        print(f'Tổng thời gian trainning: {total_time:.2f} giây')
        return hist

    def predict(self, test_data, scaler, data_scaled=True):
        '''
            Input: test_data - list of input values (numpy array) and target values
                               (numpy array) of validation data
                   scaler - scaler object to inversely scale predictions
                   data_scaled - if scaler were used in the preprocessing (boolean)

            Output: predictions - numpy array of the predicted values
        '''
        self.x_test, self.y_test = test_data
        self.x_test = self.x_test.to(device)#Chuyển dữ liệu x_test sang GPU
        self.y_test = self.y_test.to(device)#Chuyển dữ liệu y_test sang GPU
        self.model.eval()
        start_time = time.time()
        predictions = self.model(self.x_test).detach().cpu().numpy()
        end_time = time.time()
        inference_time = end_time - start_time
        print("Thời gian inference: {:.4f}s".format(inference_time))
        if data_scaled:
            predictions = scaler.inverse_transform(predictions)

        return predictions
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_predictions(df, train_data_size, predictions, model_name:str):
    '''
        Input: df - dataframe of stock values
               train_data_size - length of the training data, number of elements (int)
               predictions - numpy array of the prdicted values
    '''
    colors = ['#579BF5', '#C694F6', '#F168F1']
    fig = go.Figure()
    train = df[:train_data_size]
    valid = df[train_data_size:][:-2]
    valid['Predictions'] = predictions
    y_true = valid['Adj Close'].values
    RMSE = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    
    x_train = [str(train.index[i]).split()[0] for i in range(len(train))]
    x_val = [str(valid.index[i]).split()[0] for i in range(len(valid))]
    fig.add_trace(
        go.Scatter(x=x_train, y=train['Adj Close'], mode='lines', line_color=colors[0], line_width=2,
                   name='Training data',textfont = dict(size = 20)))

    fig.add_trace(
        go.Scatter(x=x_val, y=valid['Adj Close'], mode='lines', line_color=colors[1], line_width=2,
                   name='Test data',textfont = dict(size = 20)))

    fig.add_trace(
        go.Scatter(x=x_val, y=valid['Predictions'], mode='lines', line_color=colors[2], line_width=2,
                   name='Predictions',textfont = dict(size = 20)))

    fig.update_layout(showlegend=True)
    fig.update_layout(title=dict(text=f'Predictions of stock "{train["Company stock name"][0]}" from {x_val[0]} to {x_val[len(valid) - 1]}',
    xanchor='auto', font=dict(size=24) ) ,
    xaxis=go.layout.XAxis(
    title=go.layout.xaxis.Title(
    text="Time (Day)", font=dict(size=24))),
    yaxis=go.layout.YAxis(
    title=go.layout.yaxis.Title(
    text="Adjusted closing price USD ($)", font=dict(size=24)))
    )
    fig.write_image(f'./demonstration_images/{model_name}_predictions.png')
    fig.show()











