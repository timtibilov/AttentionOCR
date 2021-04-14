import os
import PIL
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose


class TrainTestDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        formulas_path: str,
        vocab_path: str,
        transform: None
    ):
        super().__init__()

        # Loading formulas
        with open(formulas_path, 'r') as f:
            strs = f.readlines()
        data = {'tokens': [np.array(s.split()) for s in strs]}
        self.formulas = pd.DataFrame(data)

        # Loading data (image and formula index)
        self.image_dir = image_dir
        with open(data_path, 'r') as f:
            strs = f.readlines()
        strs = [s.split() for s in strs]
        data = np.array(list(zip(*strs))).T
        self.data = pd.DataFrame(data, columns=['image', 'idx'])

        self.data.drop(
            self.data[self.data.image.apply(self._check_exists)].index, inplace=True)

        self.data['idx'] = self.data['idx'].astype(int)
        self.data['image'] = self.data.image.apply(self._load_image)

        # Loading LaTeX vocabulary
        with open(vocab_path, 'r') as f:
            strs = f.readlines()
        self.vocab = {s.strip(): i for i, s in enumerate(strs)}
        self.vocab['EOF'] = len(
            self.vocab) if 'EOF' not in self.vocab else self.vocab['EOF']
        self.vocab['UNKNOWN'] = len(
            self.vocab) if 'UNKNOWN' not in self.vocab else self.vocab['UNKNOWN']
        self.vocab_size = len(self.vocab)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Getting data
        item = self.data.iloc[index]
        formula = self.formulas.iloc[item.idx].values[0]
        image = item.image
        seq_len = len(formula) + 1

        # Upsampling and transforming image
        image = self._upsample_image(image)
        if self.transform:
            image = self.transform(image)

        # Removing background from image
        image[image > 0.77] = 1.

        tokens = self._encode_tokens(formula)
        item = {
            'img': image,
            'tokens': tokens,
            'len': seq_len
        }
        return item

    def _check_exists(self, img: str) -> bool:
        return not os.path.exists(os.path.join(self.image_dir, img))

    def _load_image(self, img: str) -> Image.Image:
        """ Returns PIL.Image with 'L' convertation mode """
        im_path = os.path.join(self.image_dir, img)
        return Image.open(im_path).convert('L')

    def _encode_tokens(self, formula: list):
        """
        Encode sequence of tokens in formula.
        Returns torch.tensor of size (formula_len, vocab_len)
        """
        def encode(token):
            return self.vocab.get(token, self.vocab['UNKNOWN'])

        tokens_idx = torch.Tensor(
            list(map(encode, formula)) + [self.vocab['EOF']]).long()

        return tokens_idx

    def _upsample_image(self, image: Image.Image) -> Image.Image:
        """
        Upsample given image by random ratio in interval [1, 1.5].
        Used for case of different screen sizes. Returns upsampled image
        """
        ratio = (np.random.rand() / 2) + 1
        old_size = image.size
        new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
        image = image.resize(new_size, PIL.Image.LANCZOS)

        return image


def collate(batch, device):
    images, tokens, seq_len = zip(*batch)
    max_len = np.max(seq_len)
    tokens = [torch.cat((t, torch.zeros(max_len - t.shape[0], t.shape[1])))
              for t in tokens]
    batch = {
        'img': torch.tensor(images).to(device),
        'tokens': torch.tensor(tokens).to(device),
        'len': seq_len
    }

    return batch


def get_dataloader(
    data_path: str,
    image_dir: str,
    formulas_path: str,
    vocab_path: str,
) -> Tuple[DataLoader, Dict[str, int]]:

    transformer = Compose([
        ToTensor(),
        Normalize((0), (1))
    ])

    dataset = TrainTestDataset(
        data_path, image_dir, formulas_path, vocab_path, transformer)
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader, dataset.vocab
