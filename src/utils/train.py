import os
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch import Tensor, nn
from torch.optim import Adam
from datetime import datetime as dt
from model.model import AttentionOCR
from torch.utils.data import DataLoader
from utils.dataset import get_dataloader
from typing import Dict, Union, List, Optional
from nltk.translate.bleu_score import sentence_bleu


loss = nn.NLLLoss()


def train_epoch(dl: DataLoader, model: nn.Module, optim, device: str) -> float:
    model.train()
    batches = tqdm(dl)
    losses = []
    for b in batches:
        for k in b:
            b[k].to(device)
        optim.zero_grad()
        pred = model(b)
        curr_loss = loss(pred, b['tokens'].squeeze())

        curr_loss.backward()
        optim.step()
        losses.append(curr_loss.cpu().item())

        batches.set_description(
            f'Train epoch. Current CCE Loss: {losses[-1]}. ')
    return np.mean(losses)


def validate_epoch(dl: DataLoader, model: nn.Module, device: str) -> Dict[str, float]:
    model.eval()
    batches = tqdm(dl)
    losses = []
    bleu_scores = []

    for b in batches:
        for k in b:
            b[k].to(device)
        pred = model(b)
        curr_loss = loss(pred, b['tokens'].squeeze()).cpu().item()

        pred_tokens = torch.argmax(pred, 1).detach().cpu().numpy()
        true_tokens = b['tokens'].squeeze().cpu().numpy()
        bleu = sentence_bleu([true_tokens], pred_tokens, weights=(1,))

        losses.append(curr_loss)
        bleu_scores.append(bleu)

        batches.set_description(
            f'Test epoch. Current CCE Loss: {losses[-1]}. Current BLEU: {bleu_scores[-1]}. ')

    metrics = {
        'bleu': np.mean(bleu_scores),
        'loss': np.mean(losses)
    }

    return metrics


def fit(
    train_path: str,
    test_path: str,
    image_dir: str,
    formulas_path: str,
    vocab_path: str,
    device: Union['cpu', 'cuda'] = 'cpu',
    n_epochs: int = 12,
    lr: float = 1e-4,
    save_dir: Optional[str] = None
) -> nn.Module:

    log_file = ''.join(
        ['train_', dt.now().strftime('%Y-%m-%dT%H:%M:%S'), '.log'])
    log_path = os.path.join('./', 'logs', log_file)
    if save_dir is None:
        save_dir = os.path.join('./', 'params/')
    os.makedirs(save_dir)

    logger.add(log_path)

    logger.info('Loading train dataset')
    train_dl, vocab = get_dataloader(data_path=train_path,
                                     image_dir=image_dir,
                                     formulas_path=formulas_path,
                                     vocab_path=vocab_path,
                                     device=device)
    logger.info('Loading validation dataset')
    eval_dl, _ = get_dataloader(data_path=test_path,
                                image_dir=image_dir,
                                formulas_path=formulas_path,
                                vocab_path=vocab_path,
                                device=device)
    logger.info('Loading model')
    model = AttentionOCR(len(vocab), device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    metrics = []

    logger.info(f'Start fitting {n_epochs} epochs on {len(train_dl)} objects')
    for epoch in range(1, n_epochs):
        logger.info(f'Start {epoch} epoch of {n_epochs}')
        train_loss = train_epoch(train_dl, model, optim, device)
        logger.info(f'Train epoch {epoch}. Mean loss is {train_loss}')

        eval_metrics = validate_epoch(eval_dl, model, device)
        logger.info(
            f'Validation epoch {epoch}. Mean loss is {eval_metrics["loss"]}')
        logger.info(
            f'Validation epoch {epoch}. Mean bleu is {eval_metrics["bleu"]}')
        metrics.append(eval_metrics)
        model_name = f'{round(eval_metrics["bleu"], 3)}_{dt.now().strftime("%m-%d")}'
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        logger.info(f'Model saved at {model_path}')


def test_model():  # TODO: realize validation method
    pass
