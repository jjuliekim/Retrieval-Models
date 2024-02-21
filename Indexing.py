import os
import re

import nltk
from elasticsearch7 import Elasticsearch
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

folder = "IR_data2/IR_data/AP_DATA/ap89_collection"

nltk.download('punkt')

text_map = {}
ps = PorterStemmer()

# read stoplist.txt as a list
stopwords_path = 'config/stoplist.txt'
with open(stopwords_path) as file:
    stopwords = file.read().splitlines()


# parse the document to get ID and TEXT
def parse_file(path):
    doc_no_pattern = re.compile(r"<DOCNO>(.+?)</DOCNO>")
    text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)
    with open(path, 'r') as f:
        doc_no = ""
        content = ""
        in_text = False
        for line in f:
            if line.startswith("<DOCNO>"):
                match = doc_no_pattern.search(line)
                if match:
                    doc_no = match.group(1).strip()
            elif line.startswith("<TEXT>"):
                in_text = True
                text_match = text_pattern.search(line)
                if text_match:
                    content += text_match.group(1)
            elif in_text and "</TEXT>" not in line:
                cleaned_line = ''.join(c for c in line if c.isalnum() or c.isspace() or c == '-')
                content += cleaned_line
            elif in_text and "</TEXT>" in line:
                in_text = False
            elif line.startswith("</DOC>"):
                text_map[doc_no] = content.strip()
                doc_no = ""
                content = ""


def remove_stopwords(doc):
    updated_text = ' '.join(word.lower() for word in doc.split()
                            if word.lower() not in stopwords)
    return updated_text


def stem_text(text, ps):
    words = word_tokenize(text)
    updated = [ps.stem(word) for word in words]
    stemmed = ' '.join(updated)
    return stemmed


for filename in os.listdir(folder):
    if filename.lower() != 'readme':
        parse_file(os.path.join(folder, filename))
print('Parsing complete:', len(text_map))

for key, value in text_map.items():
    text_map[key] = stem_text(remove_stopwords(value), ps)

configurations = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 2,
        "analysis": {
            "filter": {
                "english_stop": {
                    "type": "stop",
                    "stopwords_path": stopwords
                }
            },
            "analyzer": {
                "stopped": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_stop"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "fielddata": True,
                "analyzer": "stopped",
                "index_options": "positions",
                "term_vector": "yes"
            }
        }
    }
}

es = Elasticsearch("http://localhost:9200")
print(es.ping())
index_name = "hw1_documents"

# es.indices.create(index=index_name, body=configurations)
# DeprecationWarning: The 'body' parameter is deprecated for the 'create' API
# and will be removed in a future version. Instead use API parameters directly.


def add_data(_id, text):
    es.index(index=index_name, document={'content': text}, id=_id)


i = 1
# add all docs to index
for key in text_map:
    add_data(key, text_map[key])
    print(i, key, ':', text_map[key])
    i += 1
print("documents have been added")
