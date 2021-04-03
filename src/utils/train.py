import torch
from tqdm import tqdm
from torch import Tensor, nn
from torch.optim import Adam
from typing import List, Dict
from model.model import AttentionOCR
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu


device: str = 'cpu'
loss = nn.NLLLoss()


def train_epoch(dl: DataLoader, model: nn.Module, optim) -> List[float]:
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
    return losses


def validate_epoch(dl: DataLoader, model: nn.Module) -> Dict[str, List[float]]:
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
        'bleu': bleu_scores,
        'loss': losses
    }

    return metrics


def fit():
    pass
