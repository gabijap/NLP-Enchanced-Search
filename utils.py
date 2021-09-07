import json
import os

import torch
import torch.nn.functional as F
from gensim import downloader as api
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from torch.nn.utils.rnn import pad_sequence

from params import params


def pad_collate(batch, batch_first=False, padding_value=0.0):
    """
    Pad each sequence with empty values (0). This makes every training sentence the same length.
    :param batch: list of variable length sequences
    :param batch_first:  output B x T x * if True, or T x B x * otherwise
    :param padding_value: value for padded elements
    :return: padded sequence, actual sequence lengths
    """
    seq_lens = torch.LongTensor([len(x) for x in batch])
    seq_pad = pad_sequence([torch.LongTensor(x) for x in batch], batch_first, padding_value)

    return seq_pad, seq_lens


# This is for logging results - this gets updated
results = {
    'parameters': params,
    'loss_plot_id': [],
    'loss_plot_value': [],
    'accuracy_plot_id': [],
    'MR': [],
    'CR': [],
    'MPQA': [],
    'SUBJ': [],
    'sentences_count': 0,
    'train_time': 0
}


def save_model(model_f, model_g, optim):
    """
    model: model we want to save
    optim: optimizer
    """
    checkpoint_f = {
        'state_dict': model_f.state_dict()
        # 'optimizer': optim.state_dict()
    }
    checkpoint_g = {
        'state_dict': model_g.state_dict()
        # 'optimizer': optim.state_dict()
    }
    torch.save(checkpoint_f, f'./checkpoint/{params["checkpoint"]}_model_f.pth')
    torch.save(checkpoint_g, f'./checkpoint/{params["checkpoint"]}_model_g.pth')

    with open(f'./checkpoint/{params["checkpoint"]}_results.json', 'w') as f:
        json.dump(results, f)


WORD_VECTORS = None  # Dictionary and embeddings
VOCABULARY = None  # Dictionary only (for inference, less memory required)


def load_word_vectors():
    global VOCABULARY, WORD_VECTORS
    if not WORD_VECTORS:
        print("Loading word vectors")
        if os.path.isfile(f'./checkpoint/{params["word_embeddings"]}'):
            word_vectors = KeyedVectors.load(f'./checkpoint/{params["word_embeddings"]}')
        else:
            word_vectors = api.load(params["word_embeddings"])
            word_vectors.save(f'./checkpoint/{params["word_embeddings"]}')
        print("Completed")
        WORD_VECTORS = word_vectors
        VOCABULARY = word_vectors.vocab


# Load word vectors once
load_word_vectors()


def get_vocabulary():
    return VOCABULARY


def get_word_vectors():
    return WORD_VECTORS


def get_sequence(text, vocab, max_length=360):
    sequence = []
    for (x, _) in zip(
            filter(lambda y: y in vocab, tokenize(text, lowercase=True)),
            range(max_length)
    ):
        sequence.append(vocab[x].index)
    if len(sequence) == 0:
        return [0]
    return sequence


def cosine_similarity(v, w):
    """
    Calcualtes cosine_similarity(v[i], w[j]) for all i and j.
    """
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)

    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w)

    if len(v.shape) == 1:
        v = v.unsqueeze(0)

    if len(w.shape) == 1:
        w = w.unsqueeze(0)

    v_norm = F.normalize(v, p=2, dim=1)
    w_norm = F.normalize(w, p=2, dim=1)

    return torch.mm(v_norm, w_norm.transpose(0, 1))
