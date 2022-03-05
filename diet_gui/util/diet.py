import re
from turtle import forward
import pandas as pd

import tkinter.messagebox as msgbox

class Preprocessing:
    def __init__(self, sorting):
        self.start_point = 0
        
        self.name_list = []
        self.time_list = []
        self.chat_list = []
        
        self.sorting = sorting
        
    def forward(self, file_name, start_date, end_date):
        with open(f'{file_name}', 'r', encoding='UTF-8') as f:
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
        
        return self.data_frame()

    def save_chat(self, spl):
        # 대화 내용 저장
        if ( (len(spl) == 3) 
            and (re.search(r'^사진', spl[2]) is not None) 
            and (len(spl[2]) <= 6) ):
            self.name_list.append(spl[0][1:])
            self.time_list.append(spl[1][1:])
            self.chat_list.append(spl[2][:-1])
        
    def data_frame(self):
        # 데이터 프레임 생성
        data = pd.DataFrame()
        data['Name'] = self.name_list
        data['Time'] = self.time_list
        data['Chat'] = self.chat_list
        
        # 시작 날짜 및 사진 없음 경고 
        if len(data['Chat']) == 0:
            msgbox.showwarning('저기요', '사진을 업로드한 사람이 없거나\n"시작 날짜"가 "대화 시작" 날짜보다\n이전은 아닌지 확인해주세요')
            return
    
        # 기본 - 이름순 정렬
        data = data.groupby('Name').count()
        
        # 기록 많은 순 정렬
        if self.sorting == 1:
            data = data.sort_values('Chat', ascending=False)
            return data
        
        return data
    
    
class End_date_cal:
    def __init__(self, y_2, m_2, d_2):
        self.y_2 = int(y_2)
        self.m_2 = int(m_2)
        self.d_2 = int(d_2)
        
    def forward(self):
        if self.out_of_date_warning() is not None:
            y_2, m_2, d_2 = self.cal()
            return y_2, m_2, d_2
    
    def cal(self):
        if (self.d_2 == 28) and (self.m_2 == 2):
            return self.y_2, 3, 1
        elif (self.d_2 == 30) and (self.m_2 in [4, 6, 9, 11]):
            return self.y_2, self.m_2+1, 1
        elif (self.d_2 == 31) and (self.m_2 in [1, 3, 5, 7, 8, 10]):
            return self.y_2, self.m_2+1, 1
        elif (self.d_2 == 31) and (self.m_2 == 12):
            return self.y_2+1, 1, 1
        else :
            return self.y_2, self.m_2, self.d_2 + 1
        
    def out_of_date_warning(self):
        if (self.d_2 > 28) and (self.m_2 == 2):
            msgbox.showwarning('글쎄요', '존재하지 않는 날짜입니다.')
            return
        elif (self.d_2 > 30) and (self.m_2 in [4, 6, 9, 11]):
            msgbox.showwarning('글쎄요', '존재하지 않는 날짜입니다.')
            return
        return 'OK'

