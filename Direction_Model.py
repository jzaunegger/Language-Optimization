import torch
import torch.nn.functional as F

class DirectionModel(torch.nn.Module):

    # Define the constructor function
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, seq_len):
        super(DirectionModel, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm_layer = torch.nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.flatten = torch.nn.Flatten(start_dim=1)

        mid_size = seq_len * (hidden_size * 2)

        self.linear_layer1 = torch.nn.Linear(mid_size, hidden_size)
        self.relu_act = torch.nn.ReLU()
        self.linear_layer2 = torch.nn.Linear(hidden_size, num_classes)
        self.act = torch.nn.Sigmoid()

    # Pass a input into the model
    def forward(self, input_data):

        embeddings = self.emb_layer(input_data)
        packed_outputs, (hidden,cell) = self.lstm_layer(embeddings)
        flattened_lstm_out = self.flatten(packed_outputs)
        
        dense_outputs1 = F.relu(self.linear_layer1(flattened_lstm_out))
        linear_out = self.linear_layer2(dense_outputs1)

        sig_out = torch.sigmoid(linear_out)
        sof_out = torch.softmax(linear_out, dim=1)


        return sig_out, sof_out


# Evaluate the models performance
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    predictions = []
    true_labels = []
        
    for tokens, dir_labels in loader:

        tokens = tokens.to(device)

        dir_labels = dir_labels.to(device)

        outputs, probs = model(tokens)

        for i in range(0, len(outputs)):

            out_label = torch.argmax(outputs[i]).cpu().detach().numpy()
            actual_label = torch.argmax(dir_labels[i]).cpu().detach().numpy()
                
            if out_label == actual_label: 
                correct += 1

            true_labels.append(actual_label)
            predictions.append(out_label)
                
            total += 1

    acc = correct / total
    model.train()
    return acc, predictions, true_labels


def train(model, train_loader, valid_loader, epochs, device, optimizer, logger):
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

            for tokens, dir_labels in train_loader:

                tokens = tokens.to(device)
                dir_labels  = dir_labels.to(device)

                optimizer.zero_grad()
                logits, probs = model(tokens)

                loss = loss_func(logits, dir_labels).to(device)

                loss.backward()
                optimizer.step()

            train_acc, train_preds, train_true = evaluate(model, train_loader, device)
            valid_acc, valid_preds, valid_true = evaluate(model, valid_loader, device)

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
