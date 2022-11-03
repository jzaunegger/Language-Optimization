from nltk.tokenize import word_tokenize
import torch
import os, pickle, sys

# Create a function to read a pickle file
#################################################
def load_pickle_file(filepath):

    if os.path.exists(filepath) == True:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data

    else:
        print('Error: Cannot load token data. Check path: {}'.format(filepath))
        sys.exit()

# Create a function to load in the token data
#################################################
def load_token_data(filepath):
    token_data = load_pickle_file(filepath)

    token2index = token_data['token2index']
    index2token = token_data['index2token']

    return token2index, index2token

# Function to check if a given string value could be parsed as a number
#######################################################################
def check_is_num(input_val):
    try:
        float(input_val)
        return True
    except ValueError:
        return False

# Convert index values to tokens
#######################################################################
def convert_index_seq_to_tokens(index_sequence, index2token):
    tokens = []
    for i in range(0, len(index_sequence)):

        if index_sequence in index2token:
            tokens.append(index2token[index_sequence[i]])
        else:
            print('Unknown Index: {}'.format(index_sequence[i]))
    return tokens

# Convert token values to index values
#######################################################################
def convert_token_seq_to_index(token_sequence, token2index):
    indicies = []
    for i in range(0, len(token_sequence)):
        if token_sequence[i] in token2index:
            indicies.append(token2index[token_sequence[i]])
        else:
            print('Unknown Token: {}'.format(token_sequence[i]))

    return indicies

# Function to pad a index sequence to a desired length
#######################################################################
def pad_index_sequence(index_sequence, token2index, seq_len):

    indexed_tokens = index_sequence

    num_tokens = len(indexed_tokens)
    if num_tokens < seq_len:
        diff = seq_len - num_tokens
        for i in range(0, diff):
            indexed_tokens.append(token2index['PAD'])

    elif num_tokens > seq_len:
        indexed_tokens = indexed_tokens[0:seq_len]

    return indexed_tokens


def preprocess_sentence(current_sentence, token2index, sequence_length):
    tokens = word_tokenize(current_sentence)
    raw_tokens = []
    for i in range(0, len(tokens)):
        if check_is_num(tokens[i]) == True:
            raw_tokens.append('NUMBER')
        else:
            raw_tokens.append(tokens[i])

    # Convert tokens to index and pad them, then convert to a tensor
    indexed_tokens = convert_token_seq_to_index(raw_tokens, token2index)
    padded_indicies = pad_index_sequence(indexed_tokens, token2index, sequence_length)

    return tokens, padded_indicies

# Take a sentence and predict the sentiment
#######################################################################
def analyze_sentiment(input_tensor, sent_model, sentiment_labels):
    sent_pred, sent_prob = sent_model(input_tensor)
    sent_pred_val = torch.argmax(sent_pred).cpu().detach().numpy()

    for key in sentiment_labels:
        if sent_pred_val == key:
           return {
               "sent_label": sentiment_labels[key],
               "sent_index": sent_pred_val,
               "sent_probs": sent_prob.cpu().detach().numpy()
            }

# Take a sentence and predict the goal priority
#######################################################################
def analyze_goal_priority(input_tensor, dir_model, directivity_labels):
    dir_pred, dir_prob = dir_model(input_tensor)
    dir_pred_val = torch.argmax(dir_pred).cpu().detach().numpy()

    for key in directivity_labels:
        if dir_pred_val == key:
           return {
               "gp_label": directivity_labels[key],
               "gp_index": dir_pred_val,
               "gp_probs": dir_prob.cpu().detach().numpy()
            }

# Take a sentence and predict the sentence direction
#######################################################################
def analyze_direction(input_tensor, direction_model, direction_labels):

    direction_pred, direction_prob = direction_model(input_tensor)
    direction_val = torch.argmax(direction_pred).cpu().detach().numpy()

    for key in direction_labels:
        if direction_val == key:
            return {
                "direction_label": direction_labels[key],
                "direction_index": direction_val,
                "direction_probs": direction_prob.cpu().detach().numpy()
            }

# Take a sentence and predict the parameters, values, and units
#######################################################################
def analyze_ner(input_tensor, ner_model, idx2nertype, tokens, padded_indicies, index2token):
    ner_pred = ner_model(input_tensor)

    parameters, values, units = [], [], []
    for i in range(0, len(ner_pred[0])):
        current_out = torch.argmax(ner_pred[0][i]).item()
        entity_type = idx2nertype[current_out]
        original_token = index2token[padded_indicies[i]]

        # Check if it is the number token
        if original_token == 'NUMBER':
            original_token = tokens[i]

        if entity_type == 'Parameter':
            parameters.append(original_token)

        elif entity_type == 'Units':
            units.append(original_token)

        elif entity_type == 'Value':
            values.append(original_token)
    
    if len(units) == 0:
        units.append('None')

    return {
        'parameters': parameters,
        'values': values,
        'units': units,
    }


