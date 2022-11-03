
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# Define the sentiment model
################################################################################################
class ClassifierModel(torch.nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, seq_len):
        super(ClassifierModel, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = torch.nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.flatten = torch.nn.Flatten(start_dim=1)

        mid_size = seq_len * (hidden_size * 2)

        self.linear_layer1 = torch.nn.Linear(mid_size, hidden_size)
        self.relu_act = torch.nn.ReLU()
        self.linear_layer2 = torch.nn.Linear(hidden_size, num_classes)
        self.act = torch.nn.Sigmoid()


    def evaluate(self, loader, device):
        correct, total = 0, 0
        predictions = []
        true_labels = []
        
        for tokens, sent_labels in loader:

            tokens = tokens.to(device)
            sent_labels = sent_labels.to(device)

            outputs = self.forward(tokens)

            for i in range(0, len(outputs)):
                out_label = outputs[i].cpu().detach().numpy()
                out_label = round(out_label.item())
                actual_label = int(sent_labels[i].cpu().detach().numpy())
                
                if out_label == actual_label: 
                    correct += 1

                true_labels.append(actual_label)
                predictions.append(out_label)
                
                total += 1

        acc = correct / total
        return acc, predictions, true_labels

    def forward(self, input_data):
        embeddings = self.emb_layer(input_data)
        packed_outputs, (hidden,cell) = self.lstm_layer(embeddings)
        flattened_lstm_out = self.flatten(packed_outputs)
        
        dense_outputs1 = F.relu(self.linear_layer1(flattened_lstm_out))
        out = self.linear_layer2(dense_outputs1)
        out = torch.sigmoid(out)

        return out

    def train(self, train_loader, valid_loader, epochs, device, optimizer, logger):
        epoch_labels = []
        loss_history = []
        train_acc_hist = []
        valid_acc_hist = []

        logger.add_line(' ')
        logger.add_line('Sentiment Training')
        logger.add_bar('=', 40)
        loss_func = torch.nn.BCELoss()

        for i in range(0, epochs):
            epoch_labels.append(i)

            for tokens, sent_labels in train_loader:

                tokens = tokens.to(device)
                sent_labels = sent_labels.to(device)

                optimizer.zero_grad()
                logits = self.forward(tokens)
                loss = loss_func(logits, sent_labels.unsqueeze(1)).to(device)
                loss.backward()
                optimizer.step()


            train_acc, train_preds, train_true = self.evaluate(train_loader, device)
            valid_acc, valid_preds, valid_true = self.evaluate(valid_loader, device)

            train_acc_hist.append(train_acc)
            valid_acc_hist.append(valid_acc)
            loss_history.append(loss.item())

            line = '| Epoch: {:3}/{:3} | Loss: {:.4f} | Train Acc: {:7.2%}| Validation Acc: {:7.2%}|'.format(i+1, epochs, loss.item(), train_acc, valid_acc)
            print(line)
            logger.add_line(line)

        history_data = {
            'epoch_labels': epoch_labels,
            'train_acc_hist': train_acc_hist,
            'valid_acc_hist': valid_acc_hist,
            'loss_hist': loss_history
        }

        return history_data, logger





class SentimentModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, seq_len):
        super(SentimentModel, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = torch.nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.flatten = torch.nn.Flatten(start_dim=1)

        mid_size = seq_len * (hidden_size * 2)

        self.linear_layer1 = torch.nn.Linear(mid_size, hidden_size)
        self.relu_act = torch.nn.ReLU()
        self.linear_layer2 = torch.nn.Linear(hidden_size, num_classes)
        self.act = torch.nn.Sigmoid()
    
    def evaluate(self, loader, device):

        correct, total = 0, 0
        predictions = []
        true_labels = []
        
        for tokens, dir_labels, sent_labels in loader:

            tokens = tokens.to(device)
            sent_labels = sent_labels.to(device)

            outputs = self.forward(tokens)

            for i in range(0, len(outputs)):
                out_label = outputs[i].cpu().detach().numpy()
                out_label = round(out_label.item())
                actual_label = int(sent_labels[i].cpu().detach().numpy())
                
                if out_label == actual_label: 
                    correct += 1

                true_labels.append(actual_label)
                predictions.append(out_label)
                
                total += 1

        acc = correct / total
        return acc, predictions, true_labels


    def forward(self, input_data):
        embeddings = self.emb_layer(input_data)
        packed_outputs, (hidden,cell) = self.lstm_layer(embeddings)
        flattened_lstm_out = self.flatten(packed_outputs)
        
        dense_outputs1 = F.relu(self.linear_layer1(flattened_lstm_out))
        out = self.linear_layer2(dense_outputs1)
        out = torch.sigmoid(out)

        return out

    def train(self, train_loader, valid_loader, epochs, device, optimizer, logger):
        epoch_labels = []
        loss_history = []
        train_acc_hist = []
        valid_acc_hist = []

        logger.add_line(' ')
        logger.add_line('Sentiment Training')
        logger.add_bar('=', 40)
        loss_func = torch.nn.BCELoss()

        for i in range(0, epochs):
            epoch_labels.append(i)

            for tokens, dir_labels, sent_labels in train_loader:

                tokens = tokens.to(device)
                sent_labels = sent_labels.to(device)

                optimizer.zero_grad()
                logits = self.forward(tokens)
                loss = loss_func(logits, sent_labels.unsqueeze(1)).to(device)
                loss.backward()
                optimizer.step()


            train_acc, train_preds, train_true = self.evaluate(train_loader, device)
            valid_acc, valid_preds, valid_true = self.evaluate(valid_loader, device)

            train_acc_hist.append(train_acc)
            valid_acc_hist.append(valid_acc)
            loss_history.append(loss.item())

            line = '| Epoch: {:3}/{:3} | Loss: {:.4f} | Train Acc: {:7.2%}| Validation Acc: {:7.2%}|'.format(i+1, epochs, loss.item(), train_acc, valid_acc)
            print(line)
            logger.add_line(line)

        history_data = {
            'epoch_labels': epoch_labels,
            'train_acc_hist': train_acc_hist,
            'valid_acc_hist': valid_acc_hist,
            'loss_hist': loss_history
        }

        return history_data, logger

# Define the directivity model
################################################################################################
class DirectivityModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, seq_len):
        super(DirectivityModel, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = torch.nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.flatten = torch.nn.Flatten(start_dim=1)
        mid_size = seq_len * (hidden_size * 2)
        self.linear_layer1 = torch.nn.Linear(mid_size, hidden_size)
        self.relu_act = torch.nn.ReLU()
        self.linear_layer2 = torch.nn.Linear(hidden_size, num_classes)
        self.act = torch.nn.Sigmoid()
    
    def evaluate(self, loader, device):

        correct, total = 0, 0
        predictions, true_labels = [], []
        
        for tokens, dir_labels, sent_labels in loader:

            tokens = tokens.to(device)
            dir_labels = dir_labels.to(device)

            outputs = self.forward(tokens)

            for i in range(0, len(outputs)):
                out_label = outputs[i].cpu().detach().numpy()
                out_label = round(out_label.item())
                actual_label = int(dir_labels[i].cpu().detach().numpy())
                
                if out_label == actual_label: 
                    correct += 1

                true_labels.append(actual_label)
                predictions.append(out_label)
                total += 1

        acc = correct / total
        return acc, predictions, true_labels


    def forward(self, input):
        embeddings = self.emb_layer(input)
        packed_outputs, (hidden,cell) = self.lstm_layer(embeddings)
        flattened_lstm_out = self.flatten(packed_outputs)
        dense_outputs1 = F.relu(self.linear_layer1(flattened_lstm_out))
        out = self.linear_layer2(dense_outputs1)
        out = torch.sigmoid(out)

        return out

    def train(self, train_loader, valid_loader, epochs, device, optimizer, logger):
        epoch_labels = []
        loss_history = []
        train_acc_hist = []
        valid_acc_hist = []

        logger.add_line(' ')
        logger.add_line('Directivity Training')
        logger.add_bar('=', 40)

        loss_func = torch.nn.BCELoss()

        for i in range(0, epochs):
            epoch_labels.append(i)

            for tokens, dir_labels, sent_labels in train_loader:

                tokens = tokens.to(device)
                dir_labels = dir_labels.to(device)

                optimizer.zero_grad()
                logits = self.forward(tokens)
                loss = loss_func(logits, dir_labels.unsqueeze(1)).to(device)
                loss.backward()
                optimizer.step()


            train_acc, train_preds, train_true = self.evaluate(train_loader, device)
            valid_acc, valid_preds, valid_true = self.evaluate(valid_loader, device)

            train_acc_hist.append(train_acc)
            valid_acc_hist.append(valid_acc)
            loss_history.append(loss.item())

            line = '| Epoch: {:3}/{:3} | Loss: {:.4f} | Train Acc: {:7.2%}| Validation Acc: {:7.2%}|'.format(i+1, epochs, loss.item(), train_acc, valid_acc)
            print(line)
            logger.add_line(line)

        history_data = {
            'epoch_labels': epoch_labels,
            'train_acc_hist': train_acc_hist,
            'valid_acc_hist': valid_acc_hist,
            'loss_hist': loss_history
        }

        return history_data, logger


# Define the 1st named entity recognition model
################################################################################################
class NamedEntityModel1(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(NamedEntityModel1, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = torch.nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_layer = torch.nn.Linear(2*hidden_size, num_classes)
        self.act = torch.nn.Sigmoid()

    def evaluate(self, loader, device):

        total, correct = 0, 0
        true_labels, pred_labels = [], []
        true_sent_labels, pred_sent_labels = [], []

        for input_batch, label_batch in loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = self.forward(input_batch)

            for i in range(0, len(outputs)):
                current_out_batch = outputs[i]
                current_label_batch = label_batch[i]

                temp_true, temp_pred = [], []

                for j in range(0, len(outputs[i])):
                    current_out = torch.argmax(current_out_batch[j]).item()
                    current_label = torch.argmax(current_label_batch[j]).item()

                    if current_out == current_label:
                        correct += 1
                    total += 1

                    temp_pred.append(current_out)
                    temp_true.append(current_label)
                    pred_labels.append(current_out)
                    true_labels.append(current_label)

                true_sent_labels.append(temp_true)
                pred_sent_labels.append(temp_true)

        acc = correct / total

        return acc, true_labels, pred_labels, true_sent_labels, pred_sent_labels
            

    def forward(self, input_data):
        embeddings = self.emb_layer(input_data)
        packed_outputs, (hidden,cell) = self.lstm_layer(embeddings)
        dense_out = self.linear_layer(packed_outputs)
        return dense_out

    def train(self, train_loader, valid_loader, epochs, optimizer, device, logger):
        
        epoch_labels, train_acc_hist, valid_acc_hist, loss_hist = [], [], [], []
        logger.add_line(' ')
        logger.add_line('Named Entity Recognition Training')
        logger.add_bar('=', 40)
        
        loss_func = torch.nn.CrossEntropyLoss()

        for i in range(0, epochs):
            epoch_labels.append(i)

            for input_batch, label_batch in train_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)

                optimizer.zero_grad()
                outputs = self.forward(input_batch)

                loss = loss_func(outputs, label_batch)
                loss.backward()
                optimizer.step()

                train_acc = 0
                valid_acc = 0

            train_acc, train_true_labels, train_pred_labels, train_true_sent_labels, train_pred_sent_labels = self.evaluate(train_loader, device)
            valid_acc, valid_true_labels, valid_pred_labels, valid_true_sent_labels, valid_pred_sent_labels = self.evaluate(valid_loader, device)

            train_acc_hist.append(train_acc)
            valid_acc_hist.append(valid_acc)
            loss_hist.append(loss.item())

            line = '| Epoch: {:3}/{:3} | Loss: {:.4f} | Train Acc: {:7.2%}| Validation Acc: {:7.2%}|'.format(i+1, epochs, loss.item(), train_acc, valid_acc)
            print(line)
            logger.add_line(line)
        
        history_data = {
            'epoch_labels': epoch_labels,
            'train_acc_hist': train_acc_hist,
            'valid_acc_hist': valid_acc_hist,
            'loss_hist': loss_hist
        }

        return history_data, logger

# Define the 2nd named entity recognition model
################################################################################################
class NamedEntityModel2(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(NamedEntityModel2, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embed_size)

        self.linear_1 = torch.nn.Linear(embed_size, embed_size)
        self.dropout_1 = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()
        self.lstm_layer = torch.nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout_2 = torch.nn.Dropout()

        self.linear_2 = torch.nn.Linear(2*hidden_size, num_classes)
        self.act = torch.nn.Sigmoid()

    def evaluate(self, loader, device):

        total, correct = 0, 0
        true_labels, pred_labels = [], []
        true_sent_labels, pred_sent_labels = [], []

        for input_batch, label_batch in loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = self.forward(input_batch)

            for i in range(0, len(outputs)):
                current_out_batch = outputs[i]
                current_label_batch = label_batch[i]

                temp_true, temp_pred = [], []

                for j in range(0, len(outputs[i])):
                    current_out = torch.argmax(current_out_batch[j]).item()
                    current_label = torch.argmax(current_label_batch[j]).item()

                    if current_out == current_label:
                        correct += 1
                    total += 1

                    temp_pred.append(current_out)
                    temp_true.append(current_label)
                    pred_labels.append(current_out)
                    true_labels.append(current_label)

                true_sent_labels.append(temp_true)
                pred_sent_labels.append(temp_true)

        acc = correct / total

        return acc, true_labels, pred_labels, true_sent_labels, pred_sent_labels
            

    def forward(self, input_data):
        embeddings = self.emb_layer(input_data)
        out = self.linear_1(embeddings)
        out = self.dropout_1(out)
        out = self.relu(out)
        packed_outputs, (hidden,cell) = self.lstm_layer(out)
        out = self.dropout_2(packed_outputs)
        out = self.linear_2(out)
        out = self.act(out)

        return out

    def train(self, train_loader, valid_loader, epochs, optimizer, device, logger):
        
        epoch_labels, train_acc_hist, valid_acc_hist, loss_hist = [], [], [], []
        logger.add_line(' ')
        logger.add_line('Named Entity Recognition Training')
        logger.add_bar('=', 40)
        
        loss_func = torch.nn.CrossEntropyLoss()

        for i in range(0, epochs):
            epoch_labels.append(i)

            for input_batch, label_batch in train_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)

                optimizer.zero_grad()
                outputs = self.forward(input_batch)

                loss = loss_func(outputs, label_batch)
                loss.backward()
                optimizer.step()

                train_acc = 0
                valid_acc = 0

            train_acc, train_true_labels, train_pred_labels, train_true_sent_labels, train_pred_sent_labels = self.evaluate(train_loader, device)
            valid_acc, valid_true_labels, valid_pred_labels, valid_true_sent_labels, valid_pred_sent_labels = self.evaluate(valid_loader, device)

            train_acc_hist.append(train_acc)
            valid_acc_hist.append(valid_acc)
            loss_hist.append(loss.item())

            line = '| Epoch: {:3}/{:3} | Loss: {:.4f} | Train Acc: {:7.2%}| Validation Acc: {:7.2%}|'.format(i+1, epochs, loss.item(), train_acc, valid_acc)
            print(line)
            logger.add_line(line)
        
        history_data = {
            'epoch_labels': epoch_labels,
            'train_acc_hist': train_acc_hist,
            'valid_acc_hist': valid_acc_hist,
            'loss_hist': loss_hist
        }

        return history_data, logger