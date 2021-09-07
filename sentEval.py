"""
    ***************************************************************************************
    *    This is written by referencing SentEval benchmark example code:
    *    Title: SentEval: evaluation toolkit for sentence embeddings
    *    Author: Facebook, Inc.
    *    Date: 2017
    *    Availability: https://github.com/facebookresearch/SentEval/blob/master/examples/bow.py
    *
    ***************************************************************************************
"""

import datetime
import logging
import sys
import time

import numpy as np

from quick_thoughts import QT, load_models
from utils import pad_collate, get_vocabulary, get_sequence

CHECK_POINT_F = '2021_04_05_18_36_23'
CHECK_POINT_G = '2021_04_06_11_41_59'

PATH_TO_SENTEVAL = './data/SentEval'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def prepare(params, samples):
    params.word_vec = get_vocabulary()
    params.wvec_dim = 100  # 100 or 300
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    vocab = params.word_vec

    for sent in batch:
        sent = ' '.join(sent)
        seq1 = get_sequence(sent, vocab)
        padded = pad_collate([seq1])
        qt_encoder = params['encoder']
        emb1 = qt_encoder.embedding(padded).cpu().detach().numpy()
        embeddings.append(emb1)

    embeddings = np.vstack(embeddings)
    return embeddings


def perform_evaluation(encoder_f, encoder_g, task):
    params_senteval = {'task_path': PATH_TO_SENTEVAL + '/data', 'usepytorch': True, 'kfold': 3, 'batch_size': 512,
                       'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 128,
                                      'tenacity': 3, 'epoch_size': 2}}

    encoder = QT(encoder_f, encoder_g)
    params_senteval['encoder'] = encoder
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    return se.eval([task])[task]['devacc']


if __name__ == "__main__":
    t_start = time.time()

    encoder_f, encoder_g = load_models(CHECK_POINT_F, CHECK_POINT_G)
    for dataset in ['MR', 'CR', 'MPQA', 'SUBJ', 'TREC']:  # ['MR', 'CR', 'MPQA', 'SUBJ', 'TREC']
        perform_evaluation(encoder_f, encoder_g, dataset)

    print(f'Evaluation completed in {datetime.timedelta(0, time.time() - t_start)}')
