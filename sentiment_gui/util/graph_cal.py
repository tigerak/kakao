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
        
        # print(self.dg)
        return self.fig_graph(self.dg)
        
    def my_like(self):
        self.dg.add_node(self.my_name)
        for o in self.data[self.my_name].keys():
            if self.data[self.my_name][o][5] > 1:
                self.dg.add_node(o)
                self.dg.add_edge(self.my_name, o, weight=self.data[self.my_name][o][5])

    def op_like(self):
        self.dg.add_node(self.my_name)
        for o in self.data.keys():
            for n in self.data[o].keys():
                if (n == self.my_name) and (self.data[o][n][5] > 1):
                    self.dg.add_node(o)
                    self.dg.add_edge(o, n, weight=self.data[o][n][5])
            
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
    
    def fig_graph(self, g_name):
        ax = plt.figure(figsize=(3.9, 4.3))

        if self.my_or_op == 0:
            pos = nx.spring_layout(g_name)
        else:
            pos = nx.shell_layout(g_name)
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

        plt.title('싱글이세요? 벙글인데요.')
        # plt.show()
        
        return ax