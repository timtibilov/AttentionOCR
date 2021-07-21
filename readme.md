# PDF  to LaTeX Neural Network

This neural network is based on Attention-OCR models. Install python3 and packaches before using. Type `python3 -m pip install -r requirements.txt` to install additional packages.

## Architecture

The model is based on [recent work](http://lstm.seas.harvard.edu/latex/) by Harvard University NLP Team.

<p align=center><img src='https://i.ibb.co/W6q5k33/network.png' width=200></p>

The architecture is based on Attention OCR. Contains CNN layer for feature extracting from image (encoder), and Attention + RNN Seq2Seq layer (decoder)

## Training

For training model use script `src/utils/train.py`.
Type `python3 src/utils/train.py -h` to see possible arguments.

Training process requires tokenized LaTeX formulas, images of formulas directory, LaTeX tokens vocabulary and image-formula vocabulary.

Better use small images to avoid memory stack overflow. All processes log themselves, so you can debug code easily.

## Inference

For launching Flask server execute script `src/web/app.py`.
Type `python3 src/web/app.py -h` to see possible arguments.

<p align=center><img src='https://i.ibb.co/Z80kXhP/2021-07-16-15-09-54.png' width=600></p>

Inference requires model parameters and LaTeX vocabulary. It is not allowed to load on processing images weighting 1Mb memory or more.