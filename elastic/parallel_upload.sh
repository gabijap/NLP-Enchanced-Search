#!/bin/bash

# This uploads wikipedia JSON articles to Elastic
# ./delete_index_elastic.sh wikipedia
# ./split_wiki_json.sh
# parallel ./upload_to_elastic.sh wikipedia  ::: /tmp/wikipedia/x*

# rm -f /tmp/wikipedia/x*

# This uploads arxiv JSON articles to Elastic
./delete_index_elastic.sh arxiv
./split_arxiv_json.sh
parallel ./upload_to_elastic.sh arxiv  ::: /tmp/arxiv/x*

rm -f /tmp/arxiv/x*
