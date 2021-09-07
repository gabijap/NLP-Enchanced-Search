from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)
from doc_rerank_es import rank_articles


@app.route('/')
def show_list():
    query = request.args.get('query', '')

    original_list, reranked_list = rank_articles([query])
    return render_template('list.html', query=query, original_list=original_list[:20], reranked_list=reranked_list[:20])
