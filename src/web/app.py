import os
import sys
import argparse
sys.path.insert(0, f'{os.path.join(os.path.dirname(__file__), "../")}')
from utils.inference import ModelManager
from model.cnn import CNN, ResNetCNN
from flask import Flask, render_template, url_for, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}

app = Flask(__name__)
app.secret_key = 'ML is my life'  # Debug use only!
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024
manager = None


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('index.html', message='Изображение не найдено')
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', message='Пустой файл')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                formula = manager.predict(file)
                return render_template('index.html', img=filename, formula=formula)
            else:
                return render_template('index.html', message='Неподдерживаемый формат файла')
        except RequestEntityTooLarge:
            return render_template('index.html', message='Слишком большой файл допустимый размер 256 Кб')
    else:
        return render_template('index.html', formula='E = m c ^ 2')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--cnn', type=str, default='vgg')
    parser.add_argument('--vocab', type=str,
                        default='./data/latex_vocab.txt')
    parser.add_argument('--model', type=str,
                        default='./params/0.745_04-21')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    cnn_type = ResNetCNN if args.cnn == 'resnet' else CNN
    manager = ModelManager(args.model, args.vocab, args.max_len, args.device, cnn_type)
    app.run(debug=True)
