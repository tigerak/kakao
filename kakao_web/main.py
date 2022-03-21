from flask import Blueprint, Response, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import datetime
dt = datetime.datetime.now()
time_set = dt.date()

from util import (Preprocessing, End_date_cal, Btn_my_op, 
                  Make_graph)

from __init__ import db
from data_base.talk_db import Talk_db

bp = Blueprint('main', __name__)

### Main ###
@bp.route('/')
def index():
    graph_img = Btn_my_op().ferst_show()
    
    return render_template('index.html', 
                           time_set=time_set,
                           graph_img=graph_img)

### 분석 시작 ###
UPLOAD_FOLDER = './data_base/secure_uplode'
ALLOWED_EXTENSIONS = set(['txt'])
@bp.route('/analy', methods=['GET', 'POST'])
def analy():
    if request.method == 'POST':
        s = request.form['start_date']
        e = request.form['end_date']
        n = request.form['select_anony']
        f = request.files['chat_file']
        
        # 대화 파일 저장 -> 할 필요 있나???
        file_name = os.path.join(UPLOAD_FOLDER, 'chat_file.txt') 
        f.save(file_name)
        
        # 시작 및 종료 날짜 전처리
        start_date, end_date = End_date_cal().forward(s, e)
        
        # 데이터 전처리
        global df, name_list
        df = Preprocessing().forward(file_name, start_date, end_date, n)
        
        # 데이터 베이스 입력
        Talk_db.__table__.drop(db.engine)
        Talk_db.__table__.create(db.engine)
        for index, row in df.iterrows():
            new_record = Talk_db(
                name = str(row['Name']),
                time = str(row['Time']),
                text = str(row['Chat']),
                emotion = str(row['Emotion'])
            )
            db.session.add(new_record)
        db.session.commit()
        
        name_list = sorted(df['Name'].unique())
        
    return render_template('index.html', 
                            time_set=time_set,
                            name_list=name_list)
        
### 관계 그래프 시각화 ###
@bp.route('/analy/graph', methods=['GET', 'POST'])
def graph():
    if request.method == 'POST':
        my_name = request.form['select_who']
        give = request.form['give']
        
        if give == '내가 주는 호감도':
            my_or_op = 0
        elif give == '내가 받는 호감도':
            my_or_op = 1
        elif give == '전체 호감도 분포':
            my_or_op = 2
            
        # 그래프 생성
        graph_img = Btn_my_op().forward(my_or_op, my_name, df)
    
    return render_template('index.html', 
                            time_set=time_set,
                            name_list=name_list,
                            graph_img=graph_img)