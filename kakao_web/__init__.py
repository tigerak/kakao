from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    DB_URL = 'postgresql://lcazvhah:7do_cl2oxy5l5rHneTRSaQTeZdcECpic@arjuna.db.elephantsql.com/lcazvhah'
    app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # 추가적인 메모리를 필요로 함

    db.init_app(app)
    migrate.init_app(app, db)

    import main
    app.register_blueprint(main.bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
    

