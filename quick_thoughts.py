import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from torch import from_numpy
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from params import read_parameters, params
from utils import pad_collate, get_vocabulary, get_word_vectors, get_sequence


class Encoder(nn.Module):
    def __init__(self, word_vectors, hidden_size, bidirectional=True, embeddings=False):
        super(Encoder, self).__init__()
        self.device = torch.device(params['device'])
        self.hidden_size = hidden_size
        vocabulary_size = word_vectors.vectors.shape[0]
        word_embedding_size = word_vectors.vectors.shape[1]
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=word_embedding_size)
        if embeddings:
            self.embedding.weight = nn.Parameter(from_numpy(word_vectors.vectors))
        else:
            # word_vectors.vectors[:,:] = 0  # delete GenSim word embeddings
            word_vectors.vectors = np.array(
                (1.0 - 2.0 * np.random.rand(vocabulary_size, word_embedding_size)) / 10.0, dtype=np.float32)
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size=word_embedding_size, hidden_size=self.hidden_size,
                          bidirectional=self.bidirectional)

    def forward(self, input):
        input_seq, lengths_seq = input[0].to(self.device), input[1]
        embeds = self.embedding(input_seq)
        pack_seq = pack_padded_sequence(embeds, lengths_seq, enforce_sorted=False)

        if self.bidirectional:
            hidden = torch.zeros(2, embeds.shape[1], self.hidden_size, device=self.device)
        else:
            hidden = torch.zeros(1, embeds.shape[1], self.hidden_size, device=self.device)

        _, hidden = self.gru(pack_seq, hidden)

        if self.bidirectional:
            # Forward and backward hidden states could be concatenated or added
            # return hidden[-2, :, :] + hidden[-1, :, :]
            return (hidden[-2, :, :] + hidden[-1, :, :]).squeeze()
            # return torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).squeeze()
        else:
            return hidden[-1, :, :].squeeze()


class QT:
    """
    Quick Thoughts model class
    Two encoders: for context and candidate sentences
    """
    def __init__(self, encoder_f, encoder_g):
        super(QT, self).__init__()
        self.enc_f = encoder_f
        self.enc_g = encoder_g
        self.vocabulary = get_vocabulary()
        self.device = torch.device(params['device'])

    def embedding(self, inputs):  # this takes tensor as an input
        encoding_f = self.enc_f(inputs)
        encoding_g = self.enc_g(inputs)

        # return torch.cat((encoding_f, encoding_g), dim=1)
        return torch.cat((encoding_f, encoding_g), dim=0)

    def embedding_str(self, inputs, tiny=False):  # this takes a list of stings as an input
        embed_start_time = time.time()
        corpus = DataSet(inputs, self.vocabulary)
        loader = DataLoader(corpus,
                            batch_size=1000,  # 100
                            drop_last=False,
                            # pin_memory=True,
                            collate_fn=pad_collate)

        encoding_f = torch.Tensor().to(self.device)
        encoding_g = torch.Tensor().to(self.device)
        # This supports multiple batches: batch: [tensor([[..], [..]])], encoding_f: tensor([[..],[..]])
        for i, batch in enumerate(loader):
            f = self.enc_f(batch)
            g = self.enc_g(batch)
            if len(f.shape) == 1:
                encoding_f = torch.cat((encoding_f, f.unsqueeze(0)), 0)
                encoding_g = torch.cat((encoding_g, g.unsqueeze(0)), 0)
                # encoding_g = torch.cat((encoding_g, g), 0)
            else:
                encoding_f = torch.cat((encoding_f, f), 0)
                encoding_g = torch.cat((encoding_g, g), 0)

        print(f'{len(inputs)} sentences embedded in {datetime.timedelta(0, time.time() - embed_start_time)}')
        return torch.cat((encoding_f, encoding_g), dim=1)


class DataSet(Dataset):
    """
    This is to create tensors from a list of sentence strings
    """
    def __init__(self, sentences, vocab, max_len=60):
        self.vocab = vocab
        self.max_len = max_len
        self.sentences = sentences
        print("Completed reading sentences:", len(self.sentences))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return torch.LongTensor(get_sequence(self.sentences[i], self.vocab, max_length=360))


def load_models(checkpoint_f, checkpoint_g):
    """
    checkpoint: the checkpoint prefix ('2021_02_28_14_04_19') specifying which models to load
    """
    device = torch.device(params['device'])

    if not checkpoint_g:
        checkpoint_g = checkpoint_f

    word_vectors = get_word_vectors()

    p_f = read_parameters(checkpoint_f)
    p_g = read_parameters(checkpoint_g)

    model_f = Encoder(word_vectors, p_f['hidden_dim'], p_f['bidirectional']).to(device)
    model_g = Encoder(word_vectors, p_g['hidden_dim'], p_g['bidirectional']).to(device)

    checkpoint_f = torch.load(f'./checkpoint/{checkpoint_f}_model_f.pth', map_location='cpu')  # avoid GPU RAM surge
    model_f.load_state_dict(checkpoint_f['state_dict'])

    checkpoint_g = torch.load(f'./checkpoint/{checkpoint_g}_model_g.pth', map_location='cpu')  # avoid GPU RAM surge
    model_g.load_state_dict(checkpoint_g['state_dict'])

    # For inference set requires_grad to False:
    for param in model_f.parameters():
        param.requires_grad = False
    for param in model_g.parameters():
        param.requires_grad = False

    model_f.eval()
    model_g.eval()

    return model_f, model_g
