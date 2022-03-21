from http.client import HTTP_PORT, HTTPResponse
from urllib import response
from flask import Response, make_response
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
    def __init__(self, all_chat, my_name, my_or_op):
        self.my_name = my_name
        self.my_or_op = my_or_op
        self.data = self.graph_data(all_chat)
        self.dg = nx.DiGraph()
        
        
    def forward(self):
        if self.my_or_op == 0:
            self.my_like()
        elif self.my_or_op == 1 :
            self.op_like()
        elif self.my_or_op == 2 :
            self.all_like()
        
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
            
    # 모두의 호감 관계도
    def all_like(self):
        for n in self.data.keys():
            for o in self.data[n].keys():
                if self.data[n][o][5] > 1:
                    self.dg.add_node(n)
                    self.dg.add_edge(n, o, weight=self.data[n][o][5])
    
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
        fig = plt.figure()#(figsize=(4, 4))
        plt.title('싱글이세요? 벙글인데요.')
        
        g_name = self.dg
        
        # Layout 선택
        if self.my_or_op is not 2:
            pos = nx.shell_layout(g_name)
            pos[self.my_name] = [0.0, 0.0]
        else:
            pos = nx.kamada_kawai_layout(g_name)
        
        # 노드 그리기
        if self.my_or_op is not 2:
            node_size = 2000
        else :
            d = dict(g_name.degree)
            node_size = [v * 100 for v in d.values()]
            
        nx.draw_networkx_nodes(g_name,pos,
                            node_color='green',
                            alpha = 0.1,
                            node_size=node_size)

        # 노드 라벨링
        node_labels = {}
        for node_name in g_name.nodes():
            node_labels[str(node_name)] = str(node_name)
        nx.draw_networkx_labels(g_name, pos, node_labels, 
                                font_family=font_name, 
                                font_weight='bold',
                                font_size=8)

        # 엣지 그리기
        all_weights = []
        for (node1,node2,data) in g_name.edges(data=True):
            all_weights.append(data['weight'])
        unique_weights = list(set(all_weights))

        for weight in unique_weights:
            weighted_edges = [(node1,node2,edge_attr) for (node1,node2,edge_attr) in g_name.edges(data=True) if edge_attr['weight']==weight]
            width = weight*len(node_labels.keys())*3.0/sum(all_weights)
            nx.draw_networkx_edges(g_name, pos,
                                edgelist=weighted_edges,
                                width=width,
                                edge_color='blue', alpha=0.3)
            
        # 엣지 라벨링
        if self.my_or_op is not 2:
            edge_labels = nx.get_edge_attributes(g_name, 'weight')
            nx.draw_networkx_edge_labels(g_name, pos, 
                                        edge_labels=edge_labels)
        
        return fig
    
    

### [to Me] [to There] [to All] 버튼 ###
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
import matplotlib.image as img
from io import BytesIO

class Btn_my_op:
    def init(self):
        pass
        
    def forward(self, my_or_op, my_name, all_chat):
        
        fig = Make_graph(all_chat=all_chat,
                         my_name=my_name,
                         my_or_op=my_or_op).forward()
        return self.canv(fig)
        
    # 캔버스 출력
    def canv(self, fig):
        output = BytesIO()
        canvas = FigureCanvasSVG(fig).print_svg(output)
        response = make_response(output.getvalue())
        response.content_type = 'image/svg+xml'
        output.seek(0)
        plt.savefig('static/img/graph.svg')
        plt.close()
        
        return response
        
    # 초기 그래프 출력
    def ferst_show(self):
        
        image_address = r'D:\kakao\sentiment_gui\image_data\sel_name.jpg'
        
        fig = plt.figure(figsize=(3.9, 4.285))
        plt.title('싱글이세요? 벙글인데요.')
        image = img.imread(image_address)
        plt.imshow(image)
        plt.axis('off')
        
        return self.canv(fig)
    

        