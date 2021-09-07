"""
  This is a script to convert JSON wikipedia file into single sentence per line.
"""

from gensim import utils
import nltk
import json

with utils.open('enwiki-latest.json.gz', 'rb') as f:
   for line in f:
       article = json.loads(line)

       print(article['title'], ".")
       for section_title, section_text in zip(article['section_titles'], article['section_texts']):
           print(section_title, ".")
           print(section_text)
