"""
    Script to convert arXiv dataset to format required for loading into Elasticsearch
"""

import json

new_line = "\n"
with open('arxiv-metadata-oai-snapshot.json', "r") as f:
    for line in f:
        article = json.loads(line)
        print(f'{{"index": {{"_id": "{article["id"]}"}}}}')
        print(f'{{"title": "{article["title"].replace(new_line, " ")}", "text": "{article["abstract"].replace(new_line, " ").replace("  ", " ") + article["categories"]}", "categories": "{article["categories"]}"}}')
