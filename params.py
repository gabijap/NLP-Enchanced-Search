import json
import os
import time

# This is for configuration - this does not change
params = {
    'current_dir': os.path.abspath(os.curdir),
    # tiny data set 100,000 sentences
    # 'train_dataset': './data/wikipedia/lines100000.txt',
    # quarter wiki 37.5M:
    # 'train_dataset': './data/wikipedia/wikipedia_all_cleaned-voc400000-ln160M-train-quarter.txt',
    # half wikipedia 75M:
    'train_dataset': './data/wikipedia/wikipedia_all_cleaned-voc400000-ln160M-train-half.txt',
    # complete wikipedia 150M:
    # 'train_dataset': './data/wikipedia/wikipedia_all_cleaned-voc400000-ln160M-train.txt',
    # wikipedia 15M for test:
    'test_dataset': './data/wikipedia/wikipedia_all_cleaned-voc400000-ln160M-test.txt',

    'validation_dataset': 'TBD',

    'best_checkpoint': f'2021_02_28_14_04_19',  # this is best model so far
    'checkpoint': time.strftime("%Y_%m_%d_%H_%M_%S"),

    'epochs': 1,
    'window_size': 3,
    'batch_size': 200,  # 400-15000, tried 4000, 6000
    'hidden_dim': 768,  # 768, 500-1000,  1200
    'learn_rate': 5e-4,
    'clip': 5.0,
    'bidirectional': True,
    'embeddings': True,
    'word_embeddings': 'glove-wiki-gigaword-100',  # glove-wiki-gigaword-300
    'vocabulary_size': 400000,
    'word_embedding_size': 100,
    'device': 'cpu'  # 'cuda' or 'cpu'
}


def read_parameters(checkpoint):
    with open(f'./checkpoint/{checkpoint}_results.json', "r") as params_file:
        model_params = json.load(params_file)

    return model_params['parameters']
