import datetime
import os
import time

import torch
from elasticsearch import Elasticsearch
from nltk.tokenize import sent_tokenize

from params import params
from quick_thoughts import QT, load_models
from utils import cosine_similarity

HOST = '127.0.0.1'
PORT = 6006

CHECK_POINT_F = '2021_04_05_18_36_23'
CHECK_POINT_G = '2021_04_06_11_41_59'

import nltk

nltk.download('punkt')

encoder_f, encoder_g = load_models(CHECK_POINT_F, CHECK_POINT_G)
embedder = QT(encoder_f, encoder_g)

es = Elasticsearch([{'host': HOST, 'port': PORT}], timeout=50, max_retries=10, retry_on_timeout=True)

# These are selected categories from arXiv
categories333 = [
    'AI',
    # 'AR',
    # 'CC',
    # 'CE',
    # 'CG',
    'CL',
    'CR',
    'CV',
    'CY',
    # 'DB',
    'DC',
    # 'DL',
    # 'DM',
    'DS',
    # 'ET',
    # 'FL',
    # 'GL',
    # 'GR',
    # 'GT',
    # 'HC',
    'IR',
    'IT',
    'LG',
    'LO',
    # 'MA',
    # 'MM',
    # 'MS',
    'NA',
    'NE',
    'NI',
    # 'OH',
    # 'OS',
    # 'PF',
    # 'PL',
    'RO',
    # 'SC',
    # 'SD',
    'SE',
    'SI',
    'SY'
]


def load_embeddings():
    load_start_time = time.time()
    sent_emb = torch.Tensor().to(params['device'])
    loaded_emb = torch.load(f'./checkpoint/sentence_embeddings_cs', map_location='cpu').to(params['device'])
    sent_emb = torch.cat((sent_emb, loaded_emb), 0)
    print(f'Loaded embeddings {len(sent_emb)} in {datetime.timedelta(0, time.time() - load_start_time)}')

    return sent_emb


def get_embeddings():
    global embedder, categories
    total_sentences = 0
    original_list1 = []
    articles = []
    sentences = []
    sent2art = []

    # This is a data set for semantic search
    elastic_start_time = time.time()
    res1 = es.search(index='arxiv', size=120000, body={
        "query": {
            "query_string": {
                "query": "categories:(cs.IR OR cs.AI OR cs.LG OR cs.CL)",
            }
        }
    })
    print(f'Loaded articles from Elastic in {datetime.timedelta(0, time.time() - elastic_start_time)}')

    for hit1 in res1['hits']['hits']:
        original_list1.append({'_id': hit1['_id'],
                               '_score': hit1['_score'],
                               'title': hit1['_source']['title'],
                               'text': hit1['_source']['text'],
                               'sentence': '',
                               'cos': '0'})
    print(f'len={len(original_list1)}')

    for t in original_list1:
        articles.append(t['title'] + ' . ' + t['text'])
    print("Completed reading articles:", len(articles))

    for a_idx, article in enumerate(articles):
        for sentence in sent_tokenize(article):
            sent2art.append(a_idx)
            sentences.append(sentence)

    total_sentences = total_sentences + len(sentences)
    print("Completed reading sentences:", len(sentences), total_sentences)

    if not os.path.isfile(f'./checkpoint/sentence_embeddings_cs'):
        print("Embeddings file not found, regenerating")
        sent_embeddings = embedder.embedding_str(sentences)
        torch.save(sent_embeddings, f'./checkpoint/sentence_embeddings_cs')
        del sent_embeddings
        torch.cuda.empty_cache()

    sent_embeddings = load_embeddings()

    return original_list1, articles, sentences, sent2art, sent_embeddings


original_list1, articles, sentences, sent2art, sentence_embeddings = get_embeddings()


def rank_articles(queries):
    global embedder, original_list1, articles, sentences, sent2art, sentence_embeddings

    queries_embeddings = embedder.embedding_str(queries)

    # Clear old rankings
    for d in original_list1:
        if d['sentence']:
            d['sentence'] = ''
            d['cos'] = '0'

    # Find the closest 10 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(50, len(sentences))  # take more sentences, as those might be from the same article

    # use cosine-similarity and torch.topk to find the highest 5 scores
    sim_start_time = time.time()
    cos_scores = cosine_similarity(queries_embeddings[0], sentence_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    print(f'{len(sentences)} similarities computed in {datetime.timedelta(0, time.time() - sim_start_time)}')

    print("\n\nQuery:", queries[0])
    print("\nTop 10 most similar sentences in corpus:")

    reranked_list = []
    n = 0
    for score, idx in zip(top_results[0], top_results[1]):
        print(sentences[idx], sent2art[idx], "(Score: {:.4f})".format(score))
        if not original_list1[sent2art[idx]]['sentence']:
            original_list1[sent2art[idx]]['sentence'] = sentences[idx]
            original_list1[sent2art[idx]]['cos'] = "{:.4f}".format(score)
            reranked_list.append(original_list1[sent2art[idx]])
            n = n + 1
            if n == 10:
                break

    # Original Elastic results
    res = es.search(index='arxiv', size=10, body={
        "query": {
            "simple_query_string": {
                "query": f"categories:(cs.IR OR cs.AI OR cs.LG OR cs.CL) {queries[0]}",
                "auto_generate_synonyms_phrase_query": 'false'
            }
        }
    })

    original_list2 = []
    for hit in res['hits']['hits']:
        original_list2.append({'_id': hit['_id'],
                               '_score': hit['_score'],
                               'title': hit['_source']['title'],
                               'text': hit['_source']['text'],
                               'sentence': '',
                               'cos': '0'})

    print("\nTop 1 most relevant sentence in each article with cos similarity:")
    for article2 in original_list2:
        sentences2 = sent_tokenize(article2['title'] + ' . ' + article2['text'])
        sentence_embeddings2 = embedder.embedding_str(sentences2)
        cos_scores2 = cosine_similarity(queries_embeddings[0], sentence_embeddings2)[0]
        top_results2 = torch.topk(cos_scores2, k=1)
        for score2, idx2 in zip(top_results2[0], top_results2[1]):
            article2['sentence'] = sentences2[idx2]
            article2['cos'] = "{:.4f}".format(score2)
            # print(article2['title'], article2['sentence'], article2['cos'])

    return original_list2, reranked_list


if __name__ == '__main__':
    while True:
        # query = ['off policy evaluation of the recommender systems lift .']
        print('Please enter query:')
        query = [input()]
        original_list, reranked_list = rank_articles(query)

        print("\nOriginal list (Elastic):")
        for o in original_list[0:10]:
            print(o)

        print("\nReranked list (Embeddings):")
        for p in reranked_list[0:10]:
            print(p)
