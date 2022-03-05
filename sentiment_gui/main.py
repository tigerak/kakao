from pyparsing import col
from util import Preprocessing, End_date_cal
from util import Make_graph

import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
from tkinter import *
from tkinter import filedialog

from click import progressbar

root = Tk()
root.title('Are U Alone? yes, I\'m Single')
root.geometry('800x600')
root.resizable(False, False) # 가로, 세로 크기 변경 불가

# s = ttk.Style()
# s.theme_use('clam')


### 메뉴 ###
menu = Menu(root)

menu_file = Menu(menu, tearoff=0)
menu_file.add_separator()
menu_file.add_command(label='프로그램 종료', command=root.quit)

menu_edit = Menu(menu, tearoff=0)
menu_edit.add_radiobutton(label='이름 순으로 정렬(기본)', state='disable')
menu_edit.add_radiobutton(label='횟수 순으로 정렬', state='disable')

def mail_alt():
    msgbox.showinfo('이봐요', '문제가 있나요?\ntigerakyb@gmail.com\n메일을 보내주세요 :)')
    return

menu_help = Menu(menu, tearoff=0)
menu_help.add_command(label='버그 신고', command=mail_alt)
menu_help.add_command(label='문의 메일', command=mail_alt)
menu_help.add_separator()
menu_help.add_command(label='Version_1.0')

menu.add_cascade(label='File', menu=menu_file)
menu.add_cascade(label='Edit', menu=menu_edit)
menu.add_cascade(label='Help', menu=menu_help)

root.config(menu=menu)
    
### .txt 파일 선택 ###
frame_txt_lode = LabelFrame(root, text='대화 파일 선택', bd=1)
frame_txt_lode.grid(row=0, column=0, padx=5, pady=5, sticky=N+E+W+S)

entry_text_address = Entry(frame_txt_lode)
entry_text_address.insert(0, 'txt 파일을 선택해주세요.')
entry_text_address.pack(side='left', fill='x', 
                        padx=5, pady=5,
                        expand=True, ipady=5)

def btn_txt_load_cmd():
    # 부석할 txt파일 추가 
    entry_text_address.delete(0, END)
    files = filedialog.askopenfilenames(title='.txt 파일을 선택해주세요',
                                        filetypes=(('TXT파일', '*.txt'),
                                                   ('모든 파일', '*.*')),
                                        initialdir='C:/')
    entry_text_address.insert(END, files)
    
btn_txt_load = Button(frame_txt_lode, width=10, height=1, 
                      text='찾아보기',
                      command=btn_txt_load_cmd)
btn_txt_load.pack(side='right', padx=5, pady=5, ipady=3)

### 시작 종료일 선택 ###
frame_date_lode = LabelFrame(root, text='날짜 선택', bd=1)
# frame_date_lode.pack(fill='x', padx=5, pady=5)
frame_date_lode.grid(row=1, column=0, padx=5, pady=5)

label_start = Label(frame_date_lode, text='  시작 날짜   : ')
label_end = Label(frame_date_lode, text='  종료 날짜   : ')

label_start_year = Label(frame_date_lode, text='년  ')
label_start_month = Label(frame_date_lode, text='월  ')
label_start_day = Label(frame_date_lode, text='일  ')

label_end_year = Label(frame_date_lode, text='년  ')
label_end_month = Label(frame_date_lode, text='월  ')
label_end_day = Label(frame_date_lode, text='일  ')

year_values = [str(i) for i in range(21, 25)]
month_values = [str(i) for i in range(1, 13)]
day_values = [str(i) for i in range(1, 32)]

combobox_staet_year = ttk.Combobox(frame_date_lode, width=5, height=10,
                                   values=year_values,
                                   state='readonly')
combobox_staet_month = ttk.Combobox(frame_date_lode, width=5, height=10,
                                    values=month_values,
                                    state='readonly')
combobox_staet_day = ttk.Combobox(frame_date_lode, width=5, height=10,
                                  values=day_values,
                                  state='readonly')

combobox_staet_year.current(1) # 기본 선택
combobox_staet_month.current(0)
combobox_staet_day.current(0)

combobox_end_year = ttk.Combobox(frame_date_lode, width=5, height=10,
                                   values=year_values,
                                   state='readonly')
combobox_end_month = ttk.Combobox(frame_date_lode, width=5, height=10,
                                    values=month_values,
                                    state='readonly')
combobox_end_day = ttk.Combobox(frame_date_lode, width=5, height=10,
                                  values=day_values,
                                  state='readonly')

combobox_end_year.current(1)
combobox_end_month.current(0)
combobox_end_day.current(0)

# 날짜 입력 배치
label_start.grid(row=1, column=0, padx=3, pady=5)

combobox_staet_year.grid(row=1, column=1, padx=3, pady=5)
label_start_year.grid(row=1, column=2, padx=3, pady=5)

combobox_staet_month.grid(row=1, column=3, padx=3, pady=5)
label_start_month.grid(row=1, column=4, padx=3, pady=5)

combobox_staet_day.grid(row=1, column=5, padx=3, pady=5)
label_start_day.grid(row=1, column=6, padx=3, pady=5)


label_end.grid(row=2, column=0, padx=3, pady=5)

combobox_end_year.grid(row=2, column=1, padx=3, pady=5)
label_end_year.grid(row=2, column=2, padx=3, pady=5)

combobox_end_month.grid(row=2, column=3, padx=3, pady=5)
label_end_month.grid(row=2, column=4, padx=3, pady=5)

combobox_end_day.grid(row=2, column=5, padx=5, pady=5)
label_end_day.grid(row=2, column=6, padx=3, pady=5)


### 이름 가리기 선택 ###
frame_name = LabelFrame(root, text='이름 공개 설정', bd=1)
frame_name.grid(row=2, column=0, padx=5, pady=5, sticky=N+E+W+S)

name_var = IntVar()
radio_yes_name = Radiobutton(frame_name, text='이름 공개', 
                              value=0, variable=name_var)
radio_no_name = Radiobutton(frame_name, text='이름 비공개', 
                              value=1, variable=name_var)
radio_yes_name.pack(side='left', padx=40, pady=8)
radio_no_name.pack(side='right', padx=40, pady=8)


### 분석 시작 ###

def btn_date_save_cmd():
    # 텍스트 파일 확인
    file_name = entry_text_address.get()
    
    if '.txt' not in file_name:
        msgbox.showwarning('저기요', '.txt 텍스트 파일을 추가하세요')
        return

    # 날짜 데이터 변환
    y_1 = combobox_staet_year.get()
    m_1 = combobox_staet_month.get()
    d_1 = combobox_staet_day.get()

    y_2 = combobox_end_year.get()
    m_2 = combobox_end_month.get()
    d_2 = combobox_end_day.get()
    
    try:
        e_d = End_date_cal(y_2, m_2, d_2)
        y_2, m_2, d_2 = e_d.forward()
    except:
        return
    
    start_date = str(f'--------------- 20{y_1}년 {m_1}월 {d_1}일')
    end_date = str(f'--------------- 20{y_2}년 {m_2}월 {d_2}일')
    
    # 정렬 순서
    yes_no_name = name_var.get()
    
    # 결과 분석
    global all_chat, combobox_name_values
    all_chat = Preprocessing(root,
                             progressbar_var).forward(file_name, 
                                                    start_date, 
                                                    end_date, 
                                                    yes_no_name)
    name_values = ['이름을 선택해주세요']
    for i in sorted(all_chat['Name'].unique()):
        name_values.append(i)
    combobox_name_values = ttk.Combobox(frame_graph, width=20, height=10,
                                        values=name_values,
                                        state='readonly')
    combobox_name_values.current(0) # 기본 선택
    combobox_name_values.grid(row=0, column=1, padx=5, pady=5)
            
    for i in range(7):
        e = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
        all_chat.loc[(all_chat['Emotion'] == i), 'show_motion'] = e[i]
    
    # 출력
    try: 
        # treeview 삭제
        for row in tree_result.get_children():
            tree_result.delete(row)
        # treeview 쓰기
        for i, n in enumerate(all_chat['Name']):
            tree_result.insert('', 'end', text="1", 
                            values=(n, 
                                    all_chat['Chat'][i], 
                                    all_chat['show_motion'][i]))
    except :
        return
    
# 분석 버튼
btn_date_save = Button(root, width=10, height=2, 
                      text='분석 시작',
                      command=btn_date_save_cmd)
btn_date_save.grid(row=3, column=0, padx=5, pady=5, sticky=N+E+W+S)

# 프로그래스 바 
progressbar_var = DoubleVar()
progressbar_1 = ttk.Progressbar(root, maximum=100,
                              variable=progressbar_var,
                              mode='determinate')
progressbar_1.grid(row=4, column=0, padx=5, pady=5, sticky=N+E+W+S)


### 결과 출력 ###
frame_result = LabelFrame(root, text='! 대화 감정 분석 !', bd=1)
frame_result.grid(row=5, column=0, padx=5, pady=10,
                 sticky=N+E+W+S)

# 스크롤 바
scrollbar = Scrollbar(frame_result)
scrollbar.pack(side='right', fill='y')

# Add a Treeview widget
tree_result = ttk.Treeview(frame_result, column=("c1", "c2", "c3"), 
                           yscrollcommand=scrollbar.set,
                           show='headings', height=10)

tree_result.column("# 1", anchor=W, width=40)
tree_result.heading("# 1", text="이름", anchor=W)
tree_result.column("# 2", anchor=W)
tree_result.heading("# 2", text="대화", anchor=W)
tree_result.column("# 3", anchor=W, width=20)
tree_result.heading("# 3", text="감정", anchor=W)

tree_result.pack(fill='x', padx=5, pady=5, ipadx=3, ipady=3)

#
scrollbar.config(command=tree_result.yview)


### 그래프 표현 ###
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
frame_graph = LabelFrame(root, text='! 그래프 시각화 !', 
                         bd=1, width=400)
frame_graph.grid(row=0, column=1, rowspan=6 ,padx=5, pady=5,
                 sticky=N+E+W)
    
label_graph_name = Label(frame_graph, text='  나의 이름은  :')
label_graph_name.grid(row=0, column=0, padx=5, pady=5)

# 이름 선택
name_values = ['이름을 선택해주세요']
combobox_name_values = ttk.Combobox(frame_graph, width=20, height=10,
                                values=name_values,
                                state='readonly')
combobox_name_values.current(0) # 기본 선택
combobox_name_values.grid(row=0, column=1, padx=5, pady=5)

# to me to there 버튼
def btn_my_cmd():
    my_name = combobox_name_values.get()
    fig = Make_graph(all_chat=all_chat,
                    my_name=my_name,
                    my_or_op=0).forward()
    canvas = FigureCanvasTkAgg(fig, master=frame_graph) #
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=2) 

def btn_op_cmd():
    my_name = combobox_name_values.get()
    fig = Make_graph(all_chat=all_chat,
                    my_name=my_name,
                    my_or_op=1).forward()
    canvas = FigureCanvasTkAgg(fig, master=frame_graph) #
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=2)    
    
btn_my = Button(frame_graph, width=20, height=2, 
                    text='내가 호감을 보인 사람',
                    command=btn_my_cmd)
btn_my.grid(row=1, column=0, padx=5, pady=5)

btn_op = Button(frame_graph, width=20, height=2, 
                      text='내게 호감을 보인 사람',
                      command=btn_op_cmd)
btn_op.grid(row=1, column=1, padx=5, pady=5)
    











##########################################################

root.mainloop()