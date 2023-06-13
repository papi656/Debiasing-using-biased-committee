import os 
import math 
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn.functional as F
import torch.nn.init as init

input_path = 'datasets'
output_path = 'resources'

MAX_LEN = 310 # suitable for all datasets
MAX_GRAD_NORM = 10
BATCH_SIZE = 6
LEARNING_RATE = 1e-5

def read_data(dataset_name):
    train_path = os.path.join(input_path, dataset_name, 'train.txt')
    devel_path = os.path.join(input_path, dataset_name, 'devel.txt')
    train_token_lst, train_label_lst = [], []
    with open(train_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                train_token_lst.append(math.nan)
                train_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            train_token_lst.append(a[0].strip())
            train_label_lst.append(a[1].strip())

    train_data = pd.DataFrame({'Tokens': train_token_lst, 'Labels': train_label_lst})

    devel_token_lst, devel_label_lst = [], []
    with open(devel_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                devel_token_lst.append(math.nan)
                devel_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            devel_token_lst.append(a[0].strip())
            devel_label_lst.append(a[1].strip())

    devel_data = pd.DataFrame({'Tokens': devel_token_lst, 'Labels': devel_label_lst})

    return train_data, devel_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    label_list = [x for x in label_list if not pd.isna(x)]
    # sorting as applying set operation does not maintain the order
    label_list.sort()
    id2label = {}
    for index, label in enumerate(label_list):
        id2label[index] = label
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id

def convert_to_sentence(df):
    sent = ""
    sent_list = []
    label = ""
    label_list = []
    for tok,lab in df.itertuples(index = False):
        if isinstance(tok, float):
            sent = sent[1:]
            sent_list.append(sent)
            sent = ""
            label = label[1:]
            label_list.append(label)
            label = ""
        else:
            sent = sent + " " +str(tok)
            label = label+ "," + str(lab)
    if sent != "":
        sent_list.append(sent)
        label_list.append(label)

    return sent_list,label_list

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    sentence = str(sentence).strip()
    text_labels = str(text_labels)

    for word, label in zip(sentence.split(), text_labels.split(',')):
        # tokenize and count num of subwords
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        # add same label of word to other subwords
        labels.extend([label]*n_subwords)

    return tokenized_sentence, labels 


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id, id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label
        self.maximum_across_all = 0 

    def __getitem__(self, index):
        # step 1: tokenize sentence and adapt labels
        sentence = self.data.Sentence[index]
        word_labels = self.data.Labels[index]
        label2id = self.label2id

        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

        # step 2: add special tokens and corresponding labels
        tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
        labels.insert(0, 'O')
        labels.insert(-1, 'O')
        
        # step 3: truncating or padding
        max_len = self.max_len

        if len(tokenized_sentence) > max_len:
            #truncate
            tokenized_sentence = tokenized_sentence[:max_len]
            labels = labels[:max_len]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(max_len - len(tokenized_sentence))]
            labels = labels + ['O' for _ in range(max_len - len(labels))]

        # step 4: obtain attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [label2id[label] for label in labels]

        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.int32),
            'mask': torch.tensor(attn_mask, dtype=torch.int32),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

def create_classifier(f_dim, output_dim, linear_bias=True):
    # Randomly initialize the classifier
    classifier = torch.nn.Sequential(
        torch.nn.Linear(f_dim, int(f_dim/2), bias=linear_bias),
        torch.nn.ReLU(),
        torch.nn.Linear(int(f_dim/2), output_dim, bias=linear_bias),
    )

    # Initialize the weights and biases of the linear layers
    for module in classifier.modules():
        if isinstance(module, torch.nn.Linear):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
    
    return classifier

def train(model, dataloader, classifiers, optimizers, main_clf, main_clf_optim, device, param_lambda, clf_count):
    tr_loss = 0
    main_clf_total_loss = 0
    # put bert model in eval mode as we need to train only the classifiers
    model.eval()

    for idx, batch in enumerate(dataloader):
        input_ids = batch['ids'].to(device, dtype=torch.int32)
        mask = batch['mask'].to(device, dtype=torch.int32)
        targets = batch['targets'].to(device, dtype=torch.long)

        output_bert = model(input_ids=input_ids, attention_mask=mask)
        logits = output_bert.last_hidden_state

        curr_clf = classifiers[idx % clf_count]
        curr_optim = optimizers[idx % clf_count]

        gradient_loss = 0

        output = curr_clf(logits)

        if idx >= clf_count:
            curr_grad = 0
            for j, param in enumerate(curr_clf.parameters()):
                if j == 2 and param.grad is not None:
                    curr_grad = param.grad
                    curr_grad = curr_grad.sum(0)

            for a in range(clf_count):
                if a != (idx % clf_count):
                    for j, param in enumerate(classifiers[a].parameters()):
                        if j == 2 and param.grad is not None:
                            other_grad = param.grad
                    other_grad = other_grad.sum(0)
                    gradient_loss += torch.dot(curr_grad, other_grad)

        loss = torch.nn.CrossEntropyLoss()(output.view(-1, model.config.num_labels), targets.view(-1))
        # if gradient_loss != 0:
        #     with open('test.txt', 'a') as fh:
        #         fh.write(f'{gradient_loss}\n')
        loss += gradient_loss 
        curr_optim.zero_grad()
        tr_loss += loss.item()
        loss.backward(retain_graph=True)
        curr_optim.step()

        # training of main_clf
        output_main = main_clf(logits)
        loss_main_clf = torch.nn.CrossEntropyLoss()(output_main.view(-1, model.config.num_labels), targets.view(-1))
        main_clf_total_loss += loss_main_clf.item()
        loss_main_clf.backward()
        main_clf_optim.step()

        if idx % 100 == 0:
            print(f'\tTraining loss at {idx} steps for committee : {tr_loss}')
            print(f'\tTraining loss at {idx} steps for main clf : {main_clf_total_loss}')

    print(f'\tTraining loss for committee per epoch: {tr_loss}')
    print(f'\tTraining loss for main clf per epoch: {main_clf_total_loss}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ensemble_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()

    # read data
    train_data, devel_data = read_data(args.dataset_name)
    
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    num_labels = len(id2label)

    #get list of sentence and associated label
    train_sent, train_label = convert_to_sentence(train_data)
    devel_sent,devel_label = convert_to_sentence(devel_data)

    
    #load tokenizer
    tokenizer_dir = os.path.join('resources', args.dataset_name, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    #loading model
    model_dir = os.path.join('resources', args.dataset_name, args.model_name)
    model = AutoModel.from_pretrained(model_dir, num_labels=num_labels, id2label=id2label, label2id=label2id)

    device = 'cuda' if cuda.is_available() else 'cpu'
    #loading model to device
    model.to(device)

    train_data = {'Sentence':train_sent, 'Labels':train_label}
    train_data = pd.DataFrame(train_data)
    devel_data = {'Sentence':devel_sent, 'Labels':devel_label}
    devel_data = pd.DataFrame(devel_data)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False
                    }

    devel_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True
                    }

    train_dataset = dataset(train_data, tokenizer, MAX_LEN, label2id, id2label)
    train_dataloader = DataLoader(train_dataset, **train_params)
    devel_dataset = dataset(devel_data, tokenizer, MAX_LEN, label2id, id2label)
    devel_dataloader = DataLoader(devel_dataset, **devel_params)

    # num_epochs = 20 # no reason, IEEEAccess paper used this, so trying with this number 

    classifiers = []
    for i in range(args.ensemble_size):
        classifiers.append(create_classifier(768, model.config.num_labels).to(device))

    main_clf = create_classifier(768, model.config.num_labels).to(device)
    main_clf_optim = torch.optim.Adam(main_clf.parameters(), lr=LEARNING_RATE)

    optimizers = []
    for classifier in classifiers:
        optimizers.append(torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE))    

    # warmup training
    for i in range(args.epochs):
        print(f'Epoch {i+1}:')
        train(model, train_dataloader, classifiers, optimizers, main_clf, main_clf_optim, device, 0.4, args.ensemble_size)

    # saving bias committee classifiers
    for i in range(args.ensemble_size):
        classifier = classifiers[i]
        clf_name = 'classifier_' + str(i) + '.pth'
        clf_path = os.path.join('resources', args.dataset_name, clf_name)
        torch.save(classifier.state_dict(), clf_path)

    main_clf_path = os.path.join('resources', args.dataset_name, 'classifier_main.pth')
    torch.save(main_clf.state_dict(), main_clf_path)
    

if __name__ == '__main__':
    main()