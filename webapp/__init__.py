from flask import Flask
from .views import views

def create_app():

    UPLOAD_FOLDER = '/home/user/uploads'

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'asd;lfkjqwergnqe;gljg'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    app.register_blueprint(views, url_prefix='/')

    return app 

