import os 
import math 
import pandas as pd
import argparse
from seqeval.metrics import f1_score
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
    test_path = os.path.join(input_path, dataset_name, 'test.txt')
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
    

    test_token_lst, test_label_lst = [], []
    with open(test_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                test_token_lst.append(math.nan)
                test_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            test_token_lst.append(a[0].strip())
            test_label_lst.append(a[1].strip())

    test_data = pd.DataFrame({'Tokens': test_token_lst, 'Labels': test_label_lst})

    return train_data, devel_data, test_data

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

def weighted_loss_train(model, dataloader, classifiers, optimizers, main_clf, main_clf_optim, device, param_lambda, clf_count, weight_func):
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

        weights = torch.zeros(targets.shape).to(device)
        for i in range(clf_count):
                curr_clf = classifiers[i]
                output = curr_clf(logits)
                out_soft = torch.nn.Softmax(dim=2)(output)
                out_pos = torch.argmax(out_soft, dim=2)
                tmp = (out_pos == targets).float()
                tmp = torch.tensor(tmp)
                weights += tmp

        if weight_func == 'simple_reweight':
            clf_cutoff_cnt = float(clf_count - 2)
            weights /= clf_count
            weights[weights > (clf_cutoff_cnt/clf_count)] = 0
            weights = 1 - weights

        elif weight_func == 'linear_reweight':
            weights -= clf_count
            weights = weights * 0.02
        
        else: 
            # non-linear reweight
            alpha = 1
            weights[weights >= clf_count-1] = 0
            weights = 1 / (weights + alpha)


        main_clf_logits = main_clf(logits)
        loss = torch.nn.CrossEntropyLoss(reduction='none')(main_clf_logits.view(-1, model.config.num_labels), targets.view(-1))
        

        loss = weights.view(-1) * loss 

        loss = loss.sum()

        loss.backward()
        main_clf_optim.step()

        if idx % 50 == 0:
            print(f'\tDone {idx} steps')

def inference(model, clf, dataloader, tokenizer, device, id2label):
    model.eval()
    pred_lst = []

    for batch in dataloader:
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)

        output_bert = model(input_ids=ids, attention_mask=mask)
        logits = output_bert.last_hidden_state

        output = clf(logits)

        softmax_logits = F.softmax(output, dim=2)
        inference_ids = torch.argmax(softmax_logits, dim=2)

        for i in range(ids.shape[0]):
            tmp_labels = []
            tmp_test_tokens = tokenizer.convert_ids_to_tokens(ids[i])
            tmp_label_ids = inference_ids[i]

            for index, tok in enumerate(tmp_test_tokens):
                if tok in ['[CLS]', '[SEP]', '[PAD]']:
                    continue 
                else:
                    tmp_labels.append(id2label[tmp_label_ids[index].item()])

            pred_lst.append(tmp_labels)

    return pred_lst


# def get_devel_f1(model, clf, dataloader, tokenizer, device, id2label, gold_labels, tokens):
#     dev_preds = inference(model, clf, dataloader, tokenizer, device, id2label)
#     # print(dev_preds[:10])
#     adjust_dev_preds = []
#     labels = dev_preds[0]
#     i = 1
#     j = 0
#     tmp_labels = []
#     for tok in tokens:
#         if isinstance(tok, float):
#             # if len(tmp_labels) > 0:
#             adjust_dev_preds.extend(tmp_labels)
#             tmp_labels = []
#             if i >= len(dev_preds):
#                 # print(i)
#                 break
#             labels = dev_preds[i]
#             i += 1
#             j = 0
#         elif j  < len(labels):
#             sub_words = tokenizer.tokenize(tok)
#             tmp_labels.append(labels[j])
#             j += len(sub_words)
#         else:
#                 tmp_labels.append('O')
    
#     print(adjust_dev_preds[:5])
#     print(type(adjust_dev_preds[0]))
#     gold_labels = gold_labels[:len(adjust_dev_preds)]
#     dev_f1 = f1_score([adjust_dev_preds], [gold_labels])
    
#     return dev_f1


def generate_prediction_file(pred_labels, tokens, dataset_name, tokenizer, epoch_num, weight_func):
    p_name = 'preds_clf_' + str(epoch_num+1) + 'epoch.txt'
    output_file = os.path.join('resources', dataset_name, weight_func, p_name)
    labels = pred_labels[0]
    i = 1
    j = 0
    with open(output_file, 'w') as fh:
        for tok in tokens:
            if isinstance(tok, float):
                fh.write('\n')
                if i >= len(pred_labels):
                    break 
                labels = pred_labels[i]
                i += 1
                j = 0
            elif j < len(labels):
                sub_words = tokenizer.tokenize(tok)
                fh.write(f'{tok}\t{labels[j]}\n')
                j += len(sub_words)
            else:
                fh.write(f'{tok}\tO\n')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ensemble_size', type=int, required=True)
    parser.add_argument('--weight_func', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    args = parser.parse_args()

    # read data
    train_data, devel_data, test_data = read_data(args.dataset_name)
    
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    num_labels = len(id2label)

    #get list of sentence and associated label
    train_sent, train_label = convert_to_sentence(train_data)
    devel_sent, devel_label = convert_to_sentence(devel_data)
    test_sent, test_label = convert_to_sentence(test_data)

    
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
    devel_df = {'Sentence':devel_sent, 'Labels':devel_label}
    devel_df = pd.DataFrame(devel_df)
    test_df = {'Sentence':test_sent, 'Labels':test_label}
    test_df = pd.DataFrame(test_df)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False
                    }

    test_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False
                    }

    train_dataset = dataset(train_data, tokenizer, MAX_LEN, label2id, id2label)
    train_dataloader = DataLoader(train_dataset, **train_params)
    devel_dataset = dataset(devel_df, tokenizer, MAX_LEN, label2id, id2label)
    devel_dataloader = DataLoader(devel_dataset, **train_params)
    test_dataset = dataset(test_df, tokenizer, MAX_LEN, label2id, id2label)
    test_dataloader = DataLoader(test_dataset, **test_params)


    # loading saved committee classifiers
    classifiers = []
    for i in range(args.ensemble_size):
        clf_name = 'classifier_' + str(i) + '.pth'
        clf_path = os.path.join('resources', args.dataset_name, clf_name)

        curr_classifier = create_classifier(768, model.config.num_labels)
        curr_classifier.load_state_dict(torch.load(clf_path))
        curr_classifier.to(device)
        classifiers.append(curr_classifier) 

    # loading saved main classifier
    main_clf_path = os.path.join('resources', args.dataset_name, 'classifier_main.pth')
    main_clf = create_classifier(768, model.config.num_labels)
    main_clf.load_state_dict(torch.load(main_clf_path))
    main_clf.to(device)
    main_clf_optim = torch.optim.Adam(main_clf.parameters(), lr=LEARNING_RATE)

    # writing predictions for base model
    print(f'Writing predictions from main_clf before weighted loss training')
    # pred_labels = inference(model, main_clf, test_dataloader, tokenizer, device, id2label)
    # generate_prediction_file(pred_labels, test_data['Tokens'].tolist(), args.dataset_name, tokenizer, -1, args.weight_func)
    
    optimizers = []

    for i in range(args.num_epochs):
        print(f'Epoch {i+1}:')
        weighted_loss_train(model, train_dataloader, classifiers, optimizers, main_clf, main_clf_optim, device, 0.4, args.ensemble_size, args.weight_func)
        print('\tStarting inference')
        pred_labels = inference(model, main_clf, test_dataloader, tokenizer, device, id2label)
        print(f'\tWriting predictions to file')
        generate_prediction_file(pred_labels, test_data['Tokens'].tolist(), args.dataset_name, tokenizer, i, args.weight_func)
        # dev_f1 = get_devel_f1(model, main_clf, devel_dataloader, tokenizer, device, id2label, devel_data['Labels'].tolist(), devel_data['Tokens'].tolist())
        # print(f'F1-score on dev set for epoch {i+1} is {dev_f1}\n')
        # torch.error()
        # saving main_clf after each epoch
        instance_name = 'classifier_main_' + str(i+1) + 'epoch.pth'
        main_clf_path = os.path.join('resources', args.dataset_name, args.weight_func, instance_name)
        torch.save(main_clf.state_dict(), main_clf_path)
        with open('tmp.txt', 'a') as fh:
            fh.write(f'Done epoch {i}\n')


if __name__ == '__main__':
    main()