from .bert_utile import Preprocessing, End_date_cal

import matplotlib.pyplot as plt
import networkx as nx

import matplotlib.font_manager as fm
from matplotlib import rc

from IPython.core.display import display, HTML
display(HTML("<style>.container {width:90% !important;}</style>"))

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

import warnings
warnings.filterwarnings('ignore')


### 그래프 그리기 ###
class Make_graph:
    def __init__(self, all_chat, my_name, my_or_op=0):
        self.my_name = my_name
        self.my_or_op = my_or_op
        self.data = self.graph_data(all_chat)
        self.dg = nx.DiGraph()
        
        
    def forward(self):
        if self.my_or_op == 0:
            self.my_like()
        elif self.my_or_op == 1 :
            self.op_like()
        
        return self.fig_graph()
        
    # 내가 주는 호감
    def my_like(self):
        self.dg.add_node(self.my_name)
        for o in self.data[self.my_name].keys():
            if self.data[self.my_name][o][5] > 1:
                self.dg.add_node(o)
                self.dg.add_edge(self.my_name, o, weight=self.data[self.my_name][o][5])

    # 내가 받는 호감
    def op_like(self):
        self.dg.add_node(self.my_name)
        for o in self.data.keys():
            for n in self.data[o].keys():
                if (n == self.my_name) and (self.data[o][n][5] > 1):
                    self.dg.add_node(o)
                    self.dg.add_edge(o, n, weight=self.data[o][n][5])
            
    # 그래프용 데이터 가공
    def graph_data(self, result):
        g = {}
        name = result['Name']
        for i, n in enumerate(name.unique()):
            g[n] = {}

        start = 0
        last_name = name[0]
        for i, n in enumerate(name[1:]):
            if name[i] != n:
                start = 1
                last_name = name[i]
                if last_name not in g[n]:
                    g[n][last_name] = [0,0,0,0,0,0,0]
                
                g[n][last_name][result['Emotion'][i+1]] += 1
                
            elif name[i] == n:
                if start == 0:
                    pass
                else:
                    g[n][last_name][result['Emotion'][i+1]] += 1
                    
        return g
    
    # 그래프 그리기
    def fig_graph(self):
        fig = plt.figure(figsize=(3.9, 4.285))
        plt.title('싱글이세요? 벙글인데요.')
        
        g_name = self.dg
        
        pos = nx.shell_layout(g_name)
            
        pos[self.my_name] = [0.0, 0.0]
        
        nx.draw_networkx_nodes(g_name,pos,
                            node_color='green',
                            alpha = 0.1,
                            node_size=2000)

        labels = {}
        for node_name in g_name.nodes():
            labels[str(node_name)] = str(node_name)
        nx.draw_networkx_labels(g_name, pos, labels, 
                                font_family=font_name, 
                                font_weight='bold',
                                font_size=8)

        all_weights = []
        for (node1,node2,data) in g_name.edges(data=True):
            all_weights.append(data['weight'])
        unique_weights = list(set(all_weights))

        for weight in unique_weights:
            weighted_edges = [(node1,node2,edge_attr) for (node1,node2,edge_attr) in g_name.edges(data=True) if edge_attr['weight']==weight]
            width = weight*len(labels.keys())*3.0/sum(all_weights)
            nx.draw_networkx_edges(g_name, pos,
                                edgelist=weighted_edges,
                                width=width,
                                edge_color='blue', alpha=0.3)
        
        return fig
    
    

### [to Me] or [to There] 버튼
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as img

class Btn_my_op:
    def init(self):
        pass
        
    def forward(self, my_or_op, my_name, all_chat, frame_graph):
        
        fig = Make_graph(all_chat=all_chat,
                         my_name=my_name,
                         my_or_op=my_or_op).forward()
        return self.canv(fig, frame_graph)
        
    def canv(self, fig, frame_graph):
        canvas = FigureCanvasTkAgg(fig, master=frame_graph) 
        canvas.get_tk_widget().grid(row=2, column=0, columnspan=2) 
        plt.close()
        
        return fig
        
    def ferst_show(self, frame_graph):
        
        image_address = r'D:\kakao\sentiment_gui\image_data\sel_name.jpg'
        
        fig = plt.figure(figsize=(3.9, 4.285))
        plt.title('싱글이세요? 벙글인데요.')
        image = img.imread(image_address)
        plt.imshow(image)
        plt.axis('off')
        return self.canv(fig, frame_graph)
    

### 분석 버튼 ###
import tkinter.messagebox as msgbox
import tkinter.ttk as ttk

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
                                            width=20, height=10,
                                            values=name_values,
                                            state='readonly')
        combobox_name_values.current(0) # 기본 선택
        combobox_name_values.grid(row=0, column=1, padx=5, pady=5)
        
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
        
    def t(self):
        global te
        te = '가볍게 테스트'