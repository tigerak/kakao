import pandas as pd
from sklearn.model_selection import train_test_split
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

#multiprocessing
import multiprocessing as mp 
from multiprocessing import freeze_support

import argparse
from copy import deepcopy

#GPU 사용
device = torch.device("cuda:0")

#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

chatbot_data_short = pd.read_excel('./data/short/한국어_단발성_대화_데이터셋.xlsx')
chatbot_data_continuous = pd.read_excel('./data/continuous/한국어_연속적_대화_데이터셋.xlsx')

chatbot_data_continuous.rename(columns={'Unnamed: 1' : 'Sentence'}, inplace=True)
chatbot_data_continuous.rename(columns={'Unnamed: 2' : 'Emotion'}, inplace=True)

del_list = []
n_list = []
a_list = []

for i, e in enumerate(chatbot_data_continuous['Emotion']):
    if e not in ['감정', '분노', '혐오', '중립', '놀람', '행복', '공포', '슬픔', 'ㅈ중립', '분ㄴ', '중림', 'ㅍ', 'ㄴ중립', '분', '줄']:
        del_list.append(i)
    elif e in ['감정', 'ㅍ']:
        del_list.append(i)
    elif e in ['ㅈ중립', '중림', 'ㄴ중립', '줄']:
        n_list.append(i)
    elif e in ['분ㄴ', '분']:
        a_list.append(i)

del_list.extend(n_list)
del_list.extend(a_list)
chatbot_data_continuous = chatbot_data_continuous.drop(del_list)

chatbot_data = pd.concat([chatbot_data_short.iloc[:,:2], chatbot_data_continuous.iloc[:,1:3]])

chatbot_data.loc[(chatbot_data['Emotion'] == "공포"), 'Emotion'] = 0  #공포 => 0
chatbot_data.loc[(chatbot_data['Emotion'] == "놀람"), 'Emotion'] = 1  #놀람 => 1
chatbot_data.loc[(chatbot_data['Emotion'] == "분노"), 'Emotion'] = 2  #분노 => 2
chatbot_data.loc[(chatbot_data['Emotion'] == "슬픔"), 'Emotion'] = 3  #슬픔 => 3
chatbot_data.loc[(chatbot_data['Emotion'] == "중립"), 'Emotion'] = 4  #중립 => 4
chatbot_data.loc[(chatbot_data['Emotion'] == "행복"), 'Emotion'] = 5  #행복 => 5
chatbot_data.loc[(chatbot_data['Emotion'] == "혐오"), 'Emotion'] = 6  #혐오 => 6

data_list = []
for q, label in zip(chatbot_data['Sentence'], chatbot_data['Emotion'])  :
    data = []
    data.append(q)
    data.append(str(label))

    data_list.append(data)
    
dataset_train, dataset_val = train_test_split(data_list, test_size=0.15, random_state=42)
dataset_train, dataset_test = train_test_split(dataset_train, test_size=0.10, random_state=42)

# BERT 모델에 들어가기 위한 dataset을 만들어주는 클래스
class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, args):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=args.max_len, pad=args.pad, pair=args.pair)

        self.sentences = [transform([i[args.sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[args.label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self, 
                 bert, 
                 hidden_size=768, 
                 num_classes=7,
                 dr_rate=0.5,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size, num_classes)
        if self.dr_rate:
            self.dropout = nn.Dropout(p=self.dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
def train(model, partition, optimizer, loss_fn, args):
    
    train_dataloader = DataLoader(partition['train'], 
                                  batch_size=args.batch_size, num_workers=args.num_workers)
    model.train()
    
    correct = 0
    total = 0
    train_loss = 0.0
    
    t_total = len(train_dataloader) * args.num_epochs
    warmup_step = int(t_total * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader, 0):
        optimizer.zero_grad() 
        
        # get the inputs
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        
        train_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
    train_loss = train_loss / len(train_dataloader)
    train_acc = 100 * correct / total
    return model, train_loss, train_acc

def validate(model, partition, loss_fn, args):
    
    val_dataloader = DataLoader(partition['val'], 
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    model.eval()
    
    correct = 0
    total = 0
    val_loss = 0 
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in val_dataloader:
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            
            loss = loss_fn(out, label)
            
            val_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
    val_loss = val_loss / len(val_dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def test(model, partition, args):
    
    test_dataloader = DataLoader(partition['test'], 
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    model.eval()
    
    correct = 0
    total = 0
    current_labels = []
    current_preds = []
    
    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in test_dataloader:
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            current_labels.extend(label)
            current_preds.extend(predicted)
            
    test_acc = 100 * correct / total
    return test_acc, current_labels, current_preds

def experiment(partition, bertmodel, args):
    
    model = BERTClassifier(bertmodel).to(device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    min_val_loss = np.Inf
    n_epochs_stop = 5
    epochs_no_improve = 0
    early_stop = False
    iter = 0
    
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        ts = time.time()
        model, train_loss, train_acc = train(model, partition, optimizer, loss_fn, args) 
        val_loss, val_acc = validate(model, partition, loss_fn, args) 
        te = time.time()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch, train_acc, val_acc, train_loss, val_loss, te-ts))

        if val_loss < min_val_loss:
            # Save the model
            save_model_path = f'./weight/{args.title}.pt' # {str(args.l2).split(".")[1]}-
            torch.save(model.state_dict(), save_model_path) 
            epochs_no_improve = 0
            min_val_loss = val_loss

        else:
            epochs_no_improve += 1
        iter += 1
        if epoch > 4 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            break
        else:
            continue

    test_acc, current_labels, current_preds = test(model, partition, args)
    print('')
    print('Test Accurate Score :', test_acc)
    
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc

    result['test_labels'] = current_labels
    result['test_preds'] = current_preds

    return vars(args), result

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# ====== Random seed Initialization ====== #
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

# Setting parameters
args.sent_idx = 0
args.label_idx = 1
args.max_len = 64
args.pad = True
args.pair = False

args.num_workers = 4
args.batch_size = 64

args.warmup_ratio = 0.1
args.num_epochs = 4

args.max_grad_norm = 1

args.log_interval = 100

args.learning_rate =  5e-5

data_train = BERTDataset(dataset_train, tok, args)
data_val = BERTDataset(dataset_val, tok, args)
data_test = BERTDataset(dataset_test, tok, args)

partition = {'train': data_train, 'val': data_val, 'test':data_test}

args.title = 'test_0' ### Title !! ###

if __name__ == '__main__':
    mp.freeze_support()
    setting, result = experiment(partition, bertmodel, deepcopy(args))