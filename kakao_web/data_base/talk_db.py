from __init__ import db

class Talk_db(db.Model):
    __tablename__ = 'talk_db'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    time = db.Column(db.String)
    text = db.Column(db.String)
    emotion = db.Column(db.String)