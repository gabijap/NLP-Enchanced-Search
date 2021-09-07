import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import matmul
from torch.nn.functional import log_softmax
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Corpus
from params import params
from quick_thoughts import Encoder
from sentEval import perform_evaluation
from utils import pad_collate, save_model, results, get_word_vectors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(encoder_f, encoder_g, iterator, optimizer, criterion, clip):
    encoder_f.train()
    encoder_g.train()

    target_scores = smooth_target()

    for i, batch in enumerate(iterator):
        batch_start_time = time.time()

        optimizer.zero_grad()

        emb_f = encoder_f(batch)
        emb_g = encoder_g(batch)

        score = matmul(emb_f, emb_g.t())
        mask = torch.eye(len(score), device=device, dtype=torch.bool)
        score.masked_fill_(mask, 0)

        log_scores = log_softmax(score, dim=1)
        loss = criterion(log_scores, target_scores)

        loss.backward()

        clip_grad_norm_(list(encoder_f.parameters()) + list(encoder_g.parameters()), clip)

        optimizer.step()

        if i % 30000 == 0:  # 3000
            encoder_f.eval()
            encoder_g.eval()
            results['accuracy_plot_id'].append(i)
            for dataset in ['MR']:  # ['MR', 'CR', 'MPQA', 'SUBJ']
                acc = perform_evaluation(encoder_f, encoder_g, dataset)
                results[dataset].append(acc)
                writer.add_scalar(f'Accuracy/{params["checkpoint"]}/{dataset}', acc, i)

            save_model(encoder_f, encoder_g, optimizer)
            encoder_f.train()
            encoder_g.train()

        if i % 1000 == 0:
            results['loss_plot_id'].append(i)
            results['loss_plot_value'].append(loss.item())
            writer.add_scalar(f'Loss/{params["checkpoint"]}', loss.item(), i)
            print(f'Batch {i} completed in {datetime.timedelta(0, time.time() - batch_start_time)}, loss={loss.item()}')


def smooth_target():
    ws = params['window_size']
    bs = params['batch_size']
    targets = torch.zeros(bs, bs, device=device).fill_(0.1)

    # fill first diagonal with 1
    targets += torch.diag(torch.ones(bs - abs(1), device=device), diagonal=1)

    if ws > 1:
        scales = [0.6, 0.5, 0.4, 0.3][:ws - 1]
        windows = [2, 3, 4, 5][:ws - 1]
        for window, scale in zip(windows, scales):
            a = torch.empty(bs - abs(window), dtype=torch.float32, device=device)
            a.fill_(scale)
            targets += torch.diag(a, diagonal=window)

    targets /= targets.sum(1, keepdim=True)
    return targets


if __name__ == "__main__":
    writer = SummaryWriter()

    word_vectors = get_word_vectors()
    corpus = Corpus(params['train_dataset'], word_vectors.vocab)
    loader = DataLoader(corpus,
                        batch_size=params['batch_size'],
                        drop_last=True,
                        pin_memory=True,
                        collate_fn=pad_collate)

    encoder_f = Encoder(word_vectors, params['hidden_dim'], bidirectional=params['bidirectional'],
                        embeddings=params['embeddings']).cuda()
    encoder_g = Encoder(word_vectors, params['hidden_dim'], bidirectional=params['bidirectional'],
                        embeddings=params['embeddings']).cuda()
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(list(encoder_f.parameters()) + list(encoder_g.parameters()))

    train_start_time = time.time()
    for epoch in range(params['epochs']):
        epoch_start_time = time.time()
        train(encoder_f, encoder_g, loader, optimizer, criterion, params['clip'])
        epoch_time = datetime.timedelta(0, time.time() - epoch_start_time)
        print(f'Epoch {epoch} completed in {epoch_time}')

    train_time = datetime.timedelta(0, time.time() - train_start_time)
    results['train_time'] = str(train_time)
    print(f'Training completed in {train_time}')

    results['sentences_count'] = len(corpus.sentences)
    save_model(encoder_f, encoder_g, optimizer)
