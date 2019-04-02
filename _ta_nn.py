
import os
import random
import torch
import numpy as np
import pandas as pd

from torch import optim
import torch.nn.functional as F
import torch.nn as nn

from collections import Counter
from _build_data import *

cutoff_var = 0.1
learning_rate = 0.003

ta_data = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trade_ai(nn.Module):
    def __init__(self, in_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.in_size = in_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = nn.Dropout(dropout)
        
        # Setup embedding layer
        self.embedding = nn.Embedding(in_size, embed_size)
        
        # Setup additional layers
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
        self.fc = nn.Linear(lstm_size, output_size)
        self.soft = nn.LogSoftmax()


    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda(), weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda())
        #hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(), weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        
        return hidden

    def forward(self, nn_input, hidden_state):
        nn_in = nn_input.long().cuda()
        embeds = self.embedding(nn_in)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        lstm_out = lstm_out[-1,:,:]
        _out = self.dropout(lstm_out)
        _out = self.fc(_out)
        logps = self.soft(_out)
        
        return logps, hidden_state

def dataloader(messages, labels, sequence_length=30, batch_size=32, shuffle=False):
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    total_sequences = len(messages)

    for ii in range(0, total_sequences, batch_size):
        batch_messages = messages[ii: ii+batch_size]
        
        # First initialize a tensor of all zeros
        batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            start_idx = max(sequence_length - len(token_tensor), 0)
            batch[start_idx:, batch_num] = token_tensor[:sequence_length]
        
        label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])
        
        yield batch, label_tensor

def process_data(ta_data, data):
    return [ta_data[i] for i in data if i in ta_data]

def train_nn(pair, train_data, sentiments):
    bow = Counter([item for l in train_data for item in l])
    print('Data dict created. Size: ' + str(len(bow)))
    #Frequency of indicators appearing
    freqs = {k: v / len(train_data) for k, v in dict(bow).items()}
    # Too rare readings
    low_cutoff = 1e-5
    # Most common readings
    high_cutoff = int(float(len(bow)) * float(cutoff_var))
    # The k most common words in the corpus. Use `high_cutoff` as the k.
    K_most_common = bow.most_common(high_cutoff)

    _most_common = []
    for i in K_most_common:
        _most_common.append(i[0])
    filtered_words = [word for word in freqs if (word not in _most_common)]

    print()
    print('Data dict updated size: ' + str(len(filtered_words)))
    print()

    ta_data = {word: i for i, word in enumerate(filtered_words, 1)}
    id2vocab = {word: i for i, word in ta_data.items()}
    filtered = [[word for word in message if word in ta_data] for message in train_data]

    balanced = {'data': [], 'result':[]}

    for idx, sentiment in enumerate(sentiments):
        message = filtered[idx]
        if len(message) != 0:
            balanced['data'].append(message)
            balanced['result'].append(sentiment) 

    token_ids = [[ta_data[word] for word in message] for message in balanced['data']]
    sentiments = balanced['result']

    model = trade_ai(len(ta_data), 10, 6, 5, dropout=0.1, lstm_layers=1)
    model.embedding.weight.data.uniform_(-1, 1)
    model = model.cuda()
    input = torch.randint(0, 1000, (5, 4), dtype=torch.int64)
    hidden = model.init_hidden(4)

    logps, _ = model.forward(input, hidden)

    split_index = int(len(token_ids)*0.7)

    train_features = token_ids[:split_index]
    valid_features = token_ids[split_index:]
    train_labels = sentiments[:split_index]
    valid_labels = sentiments[split_index:]

    print('Train features count: ' + str(len(train_features)))
    print('Train labels count: ' + str(len(train_labels)))

    text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=20, batch_size=32))) 
    model = trade_ai(len(ta_data)+1, 200, 128, 5, dropout=0.)
    hidden = model.init_hidden(text_batch.size(1))
    model = model.cuda()
    logps, hidden = model.forward(text_batch, hidden)

    #TRAINING

    model = trade_ai(len(ta_data)+1, 32, 128, 5, lstm_layers=1, dropout=0.2)
    model.embedding.weight.data.uniform_(-1, 1)
    model.to(device)

    epochs = 15
    batch_size = 20
    clip = 5
    seq_len = 20

    print_every = 50
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    train_loss_list = []
    valid_loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        print('Starting epoch {}'.format(epoch + 1))
        
        steps = 0
        for text_batch, labels in dataloader(
                train_features, train_labels, batch_size=batch_size, sequence_length=seq_len, shuffle=True):
            steps += 1
            hidden = model.init_hidden(labels.shape[0])
            
            # Set Device
            text_batch, labels = text_batch.to(device), labels.to(device)
            for each in hidden:
                each.to(device)
            
            # zero accumulated gradients
            model.zero_grad()


            # get the output from the model
            output, hidden = model.forward(text_batch, hidden)

            # calculate the loss and perform backprop
            loss = criterion(output, labels.long())
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            # TODO Implement: Train Model
            
            if steps % print_every == 0:
                model.eval()
                val_losses = []
                accuracy = []

                with torch.no_grad():
                    for text_batch, labels in dataloader(
                            train_features, train_labels, batch_size=batch_size, sequence_length=seq_len, shuffle=True):
                        text_batch, labels = text_batch.to(device), labels.to(device)
                            
                        val_h = model.init_hidden(labels.shape[0])
                        for each in val_h:
                            each.to(device)
                        output, val_h = model.forward(text_batch, val_h)
                        val_loss = criterion(output, labels.long())
                        val_losses.append(val_loss.item())
                        
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.long().view(*top_class.shape)
                        accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())
                
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Step: {}...".format(steps),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)),
                      "Accuracy: {:.6f}".format(np.mean(accuracy)))
                train_loss_list.append(loss.item())
                valid_loss_list.append(np.mean(val_losses))
                accuracy_list.append(np.mean(accuracy))
                model.train()

    chart_dict = {'Train Loss': train_loss_list, 'Validation Loss': valid_loss_list, 'Accuracy': accuracy_list}
    train_df = pd.DataFrame(chart_dict)

    return ta_data, model, np.mean(accuracy), np.mean(val_losses), train_df

def _input_data(pair, ta_data, data_df):
    _in = create_input_data(data_df)
    tokens = process_data(ta_data, _in)
    #print('Tokens: ' + str(tokens))
    #print('Input: ' + str(_in))
    return tokens

def predict(model, data):
    model.eval()
    model.to(device)

    _input = torch.tensor(data).view(-1,1)
    # Get the NN output
    hidden = model.init_hidden(_input.size(1))
    logps, _ = model(_input, hidden)
    # Take the exponent of the NN output to get a range of 0 to 1 for each label.
    pred = logps.exp()
    preds = pred.cpu().detach().numpy()

    predictions = {'Very Bearish': round(float(preds[0][0]), 4),
    'Bearish': round(float(preds[0][1]), 4),
    'Neutral': round(float(preds[0][2]), 4),
    'Bullish': round(float(preds[0][3]), 4),
    'Very Bullish': round(float(preds[0][4]), 4)}

    predict = max(predictions, key=predictions.get)
    confidence = predictions[predict]
    if (predict == 'Very Bullish') or predict == 'Bullish':
        confidence = predictions['Bullish'] + predictions['Very Bullish']
    elif (predict == 'Very Bearish') or predict == 'Bearish':
        confidence = predictions['Bearish'] + predictions['Very Bearish']

    return predict, confidence, preds

def get_prediction(ta_data, model, pair, data_df):
    print('Making new prediction')
    _input = _input_data(pair, ta_data, data_df)
    prediction, accuracy, tensor = predict(model, _input)
    return prediction, accuracy, tensor


def test_train_nn(train_data, sentiments, vocab_cut, lr):
    bow = Counter([item for l in train_data for item in l])
    print('Data dict created. Size: ' + str(len(bow)))
    #Frequency of words appearing in message
    freqs = {k: v / len(train_data) for k, v in dict(bow).items()}
    # Too rare readings
    low_cutoff = 1e-5
    # Most common readings
    high_cutoff = int(float(len(bow)) * float(vocab_cut))
    # The k most common words in the corpus. Use `high_cutoff` as the k.
    K_most_common = bow.most_common(high_cutoff)

    #print('Most common indicators: ')

    _most_common = []
    for i in K_most_common:
        #print(i[0])
        _most_common.append(i[0])
    filtered_words = [word for word in freqs if (word not in _most_common)]

    print()
    print('Data dict updated size: ' + str(len(filtered_words)))
    print()

    ta_data = {word: i for i, word in enumerate(filtered_words, 1)}
    id2vocab = {word: i for i, word in ta_data.items()}
    filtered = [[word for word in message if word in ta_data] for message in train_data]

    balanced = {'data': [], 'result':[]}

    for idx, sentiment in enumerate(sentiments):
        message = filtered[idx]
        if len(message) != 0:
            balanced['data'].append(message)
            balanced['result'].append(sentiment) 

    token_ids = [[ta_data[word] for word in message] for message in balanced['data']]
    sentiments = balanced['result']

    model = trade_ai(len(ta_data), 10, 6, 5, dropout=0.1, lstm_layers=1)
    model.embedding.weight.data.uniform_(-1, 1)
    model = model.cuda()
    input = torch.randint(0, 1000, (5, 4), dtype=torch.int64)
    hidden = model.init_hidden(4)

    logps, _ = model.forward(input, hidden)

    split_index = int(len(token_ids)*0.7)

    train_features = token_ids[:split_index]
    valid_features = token_ids[split_index:]
    train_labels = sentiments[:split_index]
    valid_labels = sentiments[split_index:]

    print('Train features count: ' + str(len(train_features)))
    print('Train labels count: ' + str(len(train_labels)))

    text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=20, batch_size=32))) 
    model = trade_ai(len(ta_data)+1, 200, 128, 5, dropout=0.)
    hidden = model.init_hidden(text_batch.size(1))
    model = model.cuda()
    logps, hidden = model.forward(text_batch, hidden)

    #TRAINING

    model = trade_ai(len(ta_data)+1, 32, 128, 5, lstm_layers=1, dropout=0.2)
    model.embedding.weight.data.uniform_(-1, 1)
    model.to(device)

    epochs = 15
    batch_size = 20
    clip = 5
    seq_len = 20

    print_every = 50
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    train_loss_list = []
    valid_loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        print('Starting epoch {}'.format(epoch + 1))
        
        steps = 0
        for text_batch, labels in dataloader(
                train_features, train_labels, batch_size=batch_size, sequence_length=seq_len, shuffle=True):
            steps += 1
            hidden = model.init_hidden(labels.shape[0])
            
            # Set Device
            text_batch, labels = text_batch.to(device), labels.to(device)
            for each in hidden:
                each.to(device)
            
            # zero accumulated gradients
            model.zero_grad()


            # get the output from the model
            output, hidden = model.forward(text_batch, hidden)

            # calculate the loss and perform backprop
            loss = criterion(output, labels.long())
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            # TODO Implement: Train Model
            
            if steps % print_every == 0:
                model.eval()
                val_losses = []
                accuracy = []

                with torch.no_grad():
                    for text_batch, labels in dataloader(
                            train_features, train_labels, batch_size=batch_size, sequence_length=seq_len, shuffle=True):
                        text_batch, labels = text_batch.to(device), labels.to(device)
                            
                        val_h = model.init_hidden(labels.shape[0])
                        for each in val_h:
                            each.to(device)
                        output, val_h = model.forward(text_batch, val_h)
                        val_loss = criterion(output, labels.long())
                        val_losses.append(val_loss.item())
                        
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.long().view(*top_class.shape)
                        accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())
                
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Step: {}...".format(steps),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)),
                      "Accuracy: {:.6f}".format(np.mean(accuracy)))
                train_loss_list.append(loss.item())
                valid_loss_list.append(np.mean(val_losses))
                accuracy_list.append(np.mean(accuracy))
                model.train()

    chart_dict = {'Train Loss': train_loss_list, 'Validation Loss': valid_loss_list, 'Accuracy': accuracy_list}
    train_df = pd.DataFrame(chart_dict)

    return ta_data, model, np.mean(accuracy), np.mean(val_losses), train_df
