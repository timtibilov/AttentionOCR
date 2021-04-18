import os
import sys
import argparse
sys.path.insert(0, f'{os.path.join(os.path.dirname(__file__), "../")}')
from utils.inference import ModelManager
from flask import Flask, render_template, url_for, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}
MAX_FILE_SIZE = 2 ** 24

app = Flask(__name__)
app.secret_key = 'ML is my life'  # Debug use only!
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
manager = None


def allowed_file(filename):
    return '.' not in filename or \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('main.html', message='Изображение не найдено')
        file = request.files['file']
        if file.filename == '':
            return render_template('main.html', message='Пустой файл')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            formula = manager.predict(file)
            return render_template('main.html', img=filename, formula=formula)
        else:
            return render_template('main.html', message='Неподдерживаемый формат файла')
    else:
        return render_template('main.html', formula='E = mc^2', img='main.png')


@app.route('/uploads/<filename>')
def send_file(filename): 
    return send_from_directory(UPLOAD_FOLDER, filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--vocab', type=str,
                        default='./data/latex_vocab.txt')
    parser.add_argument('--model', type=str,
                        default='./params/params_0.53')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    manager = ModelManager(args.model, args.vocab, args.max_len, args.device)
    app.run(debug=True)