"""
  This is for cleaning wikipedia files:
  1. One sentence per line
  2. Lower case (only lower case words exist in vocabulary)
  3. Splitting sentences to words (like mr. will be changed to mr, as only mr exists in vocabulary),
     This will also remove the punctuation marks, as they are not a words according by gensim tokenize(),
     although are present in vocabulary.
  4. only leaving words that are in 400,000 gensim glove vocabulary.
  Usage: cd ~/src
         python clean_data.py a_books.txt
"""
import os
import sys

import nltk as nltk
from gensim import downloader as api
from gensim.models import KeyedVectors
from gensim.utils import tokenize

nltk.download('punkt')

print("Loading word vectors")
if os.path.isfile('/tmp/glove-wiki-gigaword-100'):
    word_vectors = KeyedVectors.load('/tmp/glove-wiki-gigaword-100')
else:
    word_vectors = api.load('glove-wiki-gigaword-100')
    word_vectors.save(f'/tmp/glove-wiki-gigaword-100')
print("Completed")

file_name = sys.argv[1]

source_name = file_name
target_name = "../cleaned/" + file_name + "_voc400000.txt"

print("Start cleaning files: source: ", source_name, ". target: ", target_name)

vocab = word_vectors.vocab

with open(source_name) as read_file, open(target_name, "w+") as write_file:
    for line in read_file:
        for sentence in nltk.sent_tokenize(line):
            filtered = filter(lambda y: y in vocab, tokenize(sentence, lowercase=True))

            result = ''
            for s in filtered:
                result = result + s + ' '

            if len(result.split()) > 0:  # Remove empty lines or short sentences with less than 3 words.
                write_file.write(result + '.\n')  # End line

print("Data file cleaning is completed.")
