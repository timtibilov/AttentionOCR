# PDF  to LaTeX Neural Network

This neural network is based on Attention-OCR models. Install python3 and packaches before using. Type `python3 -m pip install -r requirements.txt` to install additional packages.

### Training

For training model use script `src/utils/train.py`.
Type `python3 src/utils/train.py -h` to see possible arguments.

### Inference

For launching Flask server execute script `src/web/app.py`.
Type `python3 src/web/app.py -h` to see possible arguments.