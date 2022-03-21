import re
import pandas as pd

import tkinter.messagebox as msgbox

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import gluonnlp as nlp

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


### DateSet ###
class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, args):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=args.max_len, pad=args.pad, pair=args.pair)

        self.sentences = [transform([i]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i]

    def __len__(self):
        return (len(self.sentences))


### 분류기 ###
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
    


### 세팅 ###
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args("")

args.max_len = 64
args.pad = True
args.pair = False

args.batch_size = 64

### 모델 불러오기 ###
model = BERTClassifier(bertmodel).to(device)
model.load_state_dict(torch.load(r'D:\kakao\sentiment_gui\weight\bert_2.pt'))

### 토큰 불러오기 ##
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)



### 데이터 전처리 ###
from tkinter import *
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox

class Preprocessing:
    def __init__(self):
        self.start_point = 0
        self.sum_batch = 0
        
        self.name_list = []
        self.time_list = []
        self.chat_list = []
        self.emot_list = []
        
    def forward(self, file_name, start_date, end_date, yes_no_name):
        with open(file_name, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                patt = r'] '
                spl = re.split(patt, line, maxsplit=2)
                          
                if start_date in spl[0]:
                    self.start_point = 1
                elif end_date in spl[0]:
                    self.start_point = 2
                    
                if self.start_point == 1:
                    self.save_chat(spl)
                elif self.start_point == 2:
                    break
        f.close()
        
        self.analy()
        
        return self.dataframe(yes_no_name)
        
    # 채팅 저장
    def save_chat(self, spl):
        if len(spl) == 3:
            self.name_list.append(spl[0][1:])
            self.time_list.append(spl[1][1:])
            self.chat_list.append(spl[2][:-1])
        elif len(spl) == 1 and len(spl[0]) != 0:
            if len(self.name_list) != 0:
                self.name_list.append(self.name_list[-1])
                self.time_list.append(self.time_list[-1])
                self.chat_list.append(spl[0])
    
    # 데이터 프레임 생성
    def dataframe(self, yes_no_name):
        data = pd.DataFrame()
        data['Name'] = self.name_list
        data['Time'] = self.time_list
        data['Chat'] = self.chat_list
        data['Emotion'] = self.emot_list
        
        # 시작 날짜 및 사진 없음 경고 
        if len(data['Chat']) == 0:
            msgbox.showwarning('저기요', '대화한 사람이 없거나\n"시작 날짜"가 "대화 시작" 날짜보다\n이전은 아닌지 확인해주세요')
            print('시작 날짜 에러')
            return
        
        # 이름 제거
        if yes_no_name == 'no_name':
            for i, n in enumerate(data['Name'].unique()):
                data.loc[(data['Name'] == n), 'Name'] = str(i)
            return data
        
        return data

    # 감정 분석
    def analy(self):
        data_test = BERTDataset(self.chat_list, tok, args)
        data_len = BERTDataset(self.chat_list, tok, args).__len__()
        pred_list = self.test(data_test, data_len)
        
        # tensor -> numpy
        for e in pred_list:
            a = e.to('cpu').numpy()
            self.emot_list.append(a)
        
        # # 발화자의 감정 개수
        # isolation_dict = {}
        # for i, n in enumerate(data['Name']):
        #     if n not in isolation_dict.keys():
        #         isolation_dict[n] = [0,0,0, 0,0,0, 0]
                
        #     isolation_dict[n][emo[i]] += 1
            
        # return isolation_dict
    
    # 감정 분석
    def test(self, data_test, data_len):
        test_dataloader = DataLoader(data_test, 
                                    batch_size=args.batch_size) #, num_workers=self.args.num_workers) 

        model.eval()
        
        current_preds = []
        
        with torch.no_grad():
            for token_ids, valid_length, segment_ids in test_dataloader:
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length= valid_length
                out = model(token_ids, valid_length, segment_ids)
                
                _, predicted = torch.max(out.data, 1)

                current_preds.extend(predicted)
                
        return current_preds
    
    
    
### 종료 날짜 가공 ###
class End_date_cal:
    def __init__(self):
        pass
        
    def forward(self, s, e):
        s_y, s_m, s_d = self.int_date(s)
        e_y, e_m, e_d = self.int_date(e)
        e_y, e_m, e_d = self.cal(e_y, e_m, e_d)
        start_date = self.make_text(s_y, s_m, s_d)
        end_date = self.make_text(e_y, e_m, e_d)
        
        return start_date, end_date
    
    def int_date(self, d):
        date = re.split('-', d, maxsplit=2)
        year = date[0]
        month = date[1]
        day = date[2]
            
        return int(year), int(month), int(day)
    
    # 종료 날짜 가공
    def cal(self, e_y, e_m, e_d):
        if (e_d == 28) and (e_m == 2):
            return e_y, 3, 1
        elif (e_d == 30) and (e_m in [4, 6, 9, 11]):
            return e_y, e_m+1, 1
        elif (e_d == 31) and (e_m in [1, 3, 5, 7, 8, 10]):
            return e_y, e_m+1, 1
        elif (e_d == 31) and (e_m == 12):
            return e_y+1, 1, 1
        else :
            return e_y, e_m, e_d + 1
        
    def make_text(self, y, m, d):
        text_date = str(f'--------------- {y}년 {m}월 {d}일')
        
        return text_date



### 분석 버튼 ###
class Btn_data_save:
    def __init__(self, root, progressbar_var, frame_graph, tree_result):
        self.root = root
        self.progressbar_var = progressbar_var
        self.frame_graph = frame_graph
        self.tree_result = tree_result
        
    def forward(self, file_name, 
                y_1, m_1, d_1, 
                y_2, m_2, d_2,
                yes_no_name):
        
        start_date, end_date = self.preprocess(file_name, 
                                               y_1, m_1, d_1, 
                                               y_2, m_2, d_2)
        if start_date is None:
            return
        
        all_chat = Preprocessing(self.root,
                                 self.progressbar_var).forward(file_name, 
                                                               start_date, 
                                                               end_date, 
                                                               yes_no_name)
        combobox_name_values = self.sort_name(all_chat)
        self.print_out(all_chat)
        
        return all_chat, combobox_name_values
    
    # 기본 데이터 점검 및 가공
    def preprocess(self, file_name, y_1, m_1, d_1, y_2, m_2, d_2):
        # 텍스트 파일 확인
        if '.txt' not in file_name:
            msgbox.showwarning('저기요', '.txt 텍스트 파일을 추가하세요')
            print('텍스트 데이터 첨부 에러')
            return

        # 날짜 데이터 변환
        try:
            y_2, m_2, d_2 = End_date_cal(y_2, m_2, d_2).forward()
            
        except:
            print('종료 날짜 에러')
            return
        
        start_date = str(f'--------------- 20{y_1}년 {m_1}월 {d_1}일')
        end_date = str(f'--------------- 20{y_2}년 {m_2}월 {d_2}일')
        
        return start_date, end_date     
        
    # 이름 선택 콤보 박스
    def sort_name(self, all_chat):
        name_values = ['이름을 선택해주세요']
        if all_chat is not None:
            names = sorted(all_chat['Name'].unique())
            name_values.extend(names)
        combobox_name_values = ttk.Combobox(self.frame_graph, 
                                            width=15, height=10,
                                            values=name_values,
                                            state='readonly')
        combobox_name_values.current(0) # 기본 선택
        combobox_name_values.grid(row=0, column=1, 
                                  columnspan=2, 
                                  padx=5, pady=5,
                                  sticky=W+E)
        
        return combobox_name_values
    
    # 분석 결과 출력    
    def print_out(self, all_chat):
        for i in range(7):
            e = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
            all_chat.loc[(all_chat['Emotion'] == i), 'show_motion'] = e[i]
        
        # 출력
        try: 
            # treeview 삭제
            for row in self.tree_result.get_children():
                self.tree_result.delete(row)
            # treeview 쓰기
            for i, n in enumerate(all_chat['Name']):
                self.tree_result.insert('', 'end', text="1", 
                                        values=(n, 
                                                all_chat['Chat'][i], 
                                                all_chat['show_motion'][i])
                                        )
        except :
            return