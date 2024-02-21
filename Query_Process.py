import math
import json

from elasticsearch7 import Elasticsearch
from elasticsearch7.client.indices import IndicesClient
from nltk import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
es = Elasticsearch("http://localhost:9200")
ic = IndicesClient(es)
index_name = "hw1_documents"

# queries, parse, remove no., stop words, and stem them, etc.
query_path = 'IR_data2/IR_data/AP_DATA/query_desc.51-100.short3.txt'
queries = {}
with open(query_path) as file:
    for line in file:
        dot = line.find('.')
        query_number = int(line[:dot].strip())
        query_text = line[dot + 1:].strip()
        query_text = ' '.join(query_text.split(' '))
        query_text = ''.join(c for c in query_text if c.isalnum() or c.isspace() or c == '-')
        query_text = query_text.replace('-', ' ')
        queries[query_number] = query_text

# read stoplist.txt as a list
stopwords_path = 'config/stoplist.txt'
with open(stopwords_path) as file:
    stopwords = file.read().splitlines()


# remove stop words
def remove_stopwords(doc):
    updated_text = ' '.join(word.lower() for word in doc.split()
                            if word.lower() not in stopwords)
    return updated_text


# stem the text
def stem_text(text, ps):
    words = word_tokenize(text)
    updated = [ps.stem(word) for word in words]
    stemmed = ' '.join(updated)
    return stemmed


# put queries into dict {num : text}
for key, value in queries.items():
    queries[key] = stem_text(remove_stopwords(value), ps)


# ES built-in model
def run_es_builtin():
    output_file = "IR_data2/IR_data/AP_DATA/query_result_es_builtin.txt"
    with open(output_file, 'w') as file:
        for query_number, query_text in queries.items():
            es_query = {
                "query": {
                    "match": {
                        "content": query_text
                    }
                },
                "track_total_hits": True,
                "track_scores": True,
                "size": 1000,
                "_source": ["_id"],
                "explain": True
            }
            res = es.search(index=index_name, **es_query)

            rank = 1
            hits = res["hits"]["hits"]
            for i in range(len(hits)):
                hit = hits[i]
                doc_id = hit["_id"]
                score = hit["_score"]
                output_line = "%s Q0 %s %s %s Exp\n" % (query_number, doc_id, rank, score)
                file.write(output_line)
                rank += 1


# load doc lengths, if no file, initialize with empty dict
es_query = {
    "query": {
        "match_all": {}
    },
    "_source": False,
    "size": 84678
}
res = es.search(index=index_name, body=es_query)
docs = res["hits"]["hits"]
doc_ids = [hit["_id"] for hit in docs]

doc_length_file = "IR_data2/IR_data/AP_DATA/doc_length.json"
try:
    with open(doc_length_file, "r") as json_file:
        doc_length = json.load(json_file)
        total_length = sum(doc_length.values())
except (FileNotFoundError, json.decoder.JSONDecodeError):
    doc_length = {}
    total_length = 0

# find length of doc
for doc_id in doc_ids:
    if doc_id not in doc_length:
        res = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True)
        if "term_vectors" in res and "content" in res["term_vectors"]:
            term_vectors = res["term_vectors"]["content"]
            doc_size = sum(term_info["term_freq"] for term_info in term_vectors["terms"].values())
            doc_length[doc_id] = doc_size
            total_length += doc_size
        else:
            doc_length[doc_id] = 0

# write dict to json file
with open(doc_length_file, "w") as json_file:
    json.dump(doc_length, json_file, indent=2)

# calculate average length (from all docs)
avg_length = total_length / len(doc_ids)


# okapi tf
def run_okapi_tf():
    es_query = {
        "query": {
            "match_all": {}
        },
        "_source": False,
        "size": 84678
    }
    res = es.search(index=index_name, body=es_query)
    docs = res["hits"]["hits"]
    doc_ids = [hit["_id"] for hit in docs]

    output_file = "IR_data2/IR_data/AP_DATA/query_result_okapi_tf.txt"
    with (open(output_file, "w") as file):
        for query_number, query_text in queries.items():
            doc_scores = {}
            for doc_id in doc_ids:
                total_score = 0
                for query_term in query_text.split():
                    term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"])
                    if "term_vectors" in term_vectors and "content" in term_vectors["term_vectors"]:
                        terms = term_vectors["term_vectors"]["content"]["terms"]
                        tf = terms.get(query_term, {}).get("term_freq", 0)
                        denominator = 0.5 + 1.5 * (doc_length[doc_id] / avg_length)
                        score = tf / (tf + denominator) if (tf + denominator) != 0 else 0
                        total_score += score
                doc_scores[doc_id] = total_score

            ranked_docs = [item for item in sorted(doc_scores.items(),
                                                   key=lambda x: x[1], reverse=True) if item[1] > 0][:1000]
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                output_line = "%s Q0 %s %s %s Exp\n" % (query_number, doc_id, rank, score)
                file.write(output_line)


# calculate doc frequencies
doc_freqs_file = "IR_data2/IR_data/AP_DATA/doc_frequencies.json"
doc_freqs = {}

# Check if the JSON file already exists
try:
    with open(doc_freqs_file, "r") as json_file:
        doc_freqs = json.load(json_file)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    for doc_id in doc_ids:
        for query_term in queries.values():
            for term in query_term.split():
                doc_freqs[term] = doc_freqs.get(term, 0)
                if doc_freqs[term] != 0:
                    continue
                term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True)
                if "term_vectors" in term_vectors and "content" in term_vectors["term_vectors"]:
                    terms = term_vectors["term_vectors"]["content"]["terms"]
                    df = terms.get(term, {}).get("doc_freq", 0)
                    doc_freqs[term] = df

        # Save the document frequencies to a JSON file
        with open(doc_freqs_file, "w") as json_file:
            json.dump(doc_freqs, json_file, indent=2)


# tf-idf model
def run_tf_idf():
    msearch_queries = []
    for query_text in queries.values():
        msearch_queries.append({"index": index_name})
        msearch_queries.append(
            {"query":
                 {"match":
                      {"content": query_text}
                  },
             "_source": False,
             "size": 84678})

    # use multi search es api
    msearch_response = es.msearch(body=msearch_queries)
    for i, (query_number, query_text) in enumerate(queries.items()):
        output_lines = []
        # Extract hits for the current query from the msearch response
        hits = msearch_response["responses"][i].get("hits", {}).get("hits", [])
        for hit in hits:
            doc_id = hit["_id"]
            term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True,
                                          positions=True)
            total_score = 0
            for query_term in query_text.split():
                term_info = term_vectors.get("term_vectors", {}).get("content", {}).get("terms", {}).get(query_term, {})
                tf = term_info.get("term_freq", 0)
                denominator = 0.5 + 1.5 * (doc_length.get(doc_id) / avg_length)
                okapi_score = tf / (tf + denominator) if (tf + denominator) != 0 else 0
                df = doc_freqs.get(query_term)
                total_score += okapi_score * math.log(len(doc_ids) / max(df, 1))
            output_lines.append((total_score, doc_id))

        # add to file after each query
        output_file = f"IR_data2/IR_data/AP_DATA/query_result_tf_idf.txt"
        output_lines.sort(reverse=True, key=lambda x: x[0])
        with open(output_file, "a") as file:
            for rank, (score, doc_id) in enumerate(output_lines[:1000], start=1):
                ranked_line = f"{query_number} Q0 {doc_id} {rank} {score} Exp\n"
                file.write(ranked_line)


# okapi bm25
def run_okapi_bm():
    msearch_queries = []
    for query_text in queries.values():
        msearch_queries.append({"index": index_name})
        msearch_queries.append(
            {"query":
                 {"match":
                      {"content": query_text}
                  },
             "_source": False,
             "size": 84678})

    # use multi search es api
    msearch_response = es.msearch(body=msearch_queries)

    # constants
    k1 = 1.2
    k2 = 100
    b = 0.75

    for i, (query_number, query_text) in enumerate(queries.items()):
        output_lines = []
        hits = msearch_response["responses"][i].get("hits", {}).get("hits", [])
        num = 1
        for hit in hits:
            doc_id = hit["_id"]
            term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True,
                                          positions=True)
            total_score = 0
            for query_term in query_text.split():
                term_info = term_vectors.get("term_vectors", {}).get("content", {}).get("terms", {}).get(query_term, {})
                tf = term_info.get("term_freq", 0)
                df = doc_freqs.get(query_term)
                first = math.log((len(doc_ids) + 0.5) / (df + 0.5))
                second = (tf + k1 * tf) / (tf + k1 * ((1 - b) + b * (doc_length.get(doc_id) / avg_length)))
                third = (tf + k2 * tf) / (tf + k2)
                total_score += first * second * third
            output_lines.append((total_score, doc_id))
            print(query_number, num)
            num += 1

        # add results to output file
        output_file = f"IR_data2/IR_data/AP_DATA/query_result_okapi_bm2.txt"
        output_lines.sort(reverse=True, key=lambda x: x[0])
        with open(output_file, "a") as file:
            for rank, (score, doc_id) in enumerate(output_lines[:1000], start=1):
                ranked_line = f"{query_number} Q0 {doc_id} {rank} {score} Exp\n"
                file.write(ranked_line)


# get unique terms
unique_terms_file = "IR_data2/IR_data/AP_DATA/unique_terms.json"

# Check if the JSON file already exists
try:
    with open(unique_terms_file, "r") as json_file:
        unique_terms = json.load(json_file)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    unique_terms_set = set()
    for doc_id in doc_ids:
        term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True)
        if "term_vectors" in term_vectors and "content" in term_vectors["term_vectors"]:
            terms = term_vectors["term_vectors"]["content"]["terms"]
            unique_terms_set.update(terms.keys())
    unique_terms = list(unique_terms_set)

    # Save the document frequencies to a JSON file
    with open(unique_terms_file, "w") as json_file:
        json.dump(unique_terms, json_file, indent=2)


# Uni gram LM with Laplace smoothing
def run_laplace():
    v = len(unique_terms)
    msearch_queries = []

    for query_text in queries.values():
        msearch_queries.append({"index": index_name})
        msearch_queries.append({"query": {"match": {"content": query_text}}, "_source": False, "size": 84678})

    # use multi search es api
    msearch_response = es.msearch(body=msearch_queries)

    for i, (query_number, query_text) in enumerate(queries.items()):
        output_lines = []
        hits = msearch_response["responses"][i].get("hits", {}).get("hits", [])
        for hit in hits:
            doc_id = hit["_id"]
            term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True,
                                          positions=True)
            total_score = 0
            for query_term in query_text.split():
                term_info = term_vectors.get("term_vectors", {}).get("content", {}).get("terms", {}).get(query_term, {})
                tf = term_info.get("term_freq", -1000)
                score = (tf + 1) / (doc_length.get(doc_id) + v)
                if score > 0:
                    total_score += math.log(score)
                else:
                    total_score += -1000
            output_lines.append((total_score, doc_id))
        # add results to the output file
        output_file = f"IR_data2/IR_data/AP_DATA/query_result_laplace.txt"
        output_lines.sort(reverse=True, key=lambda x: x[0])
        valid_lines_written = 0
        with open(output_file, "a") as file:
            for rank, (score, doc_id) in enumerate(output_lines, start=1):
                if score != 0:
                    ranked_line = f"{query_number} Q0 {doc_id} {rank} {score} Exp\n"
                    file.write(ranked_line)
                    valid_lines_written += 1
                if valid_lines_written == 1000:
                    break


# Write corpus frequency to a JSON file
corpus_freq_file = "IR_data2/IR_data/AP_DATA/corpus_freq.json"
try:
    with open(corpus_freq_file, "r") as json_file:
        corpus_freq = json.load(json_file)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    corpus_freq = {}
    for query_number, query_term in queries.items():
        for term in query_term.split():
            corpus_freq[term] = corpus_freq.get(term, 0)
            if corpus_freq[term] != 0:
                continue
            for doc_id in doc_ids:
                term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True)
                if "term_vectors" in term_vectors and "content" in term_vectors["term_vectors"]:
                    terms = term_vectors["term_vectors"]["content"]["terms"]
                    tf = terms.get(term, {}).get("term_freq", 0)
                    corpus_freq[term] += tf
        # Save the corpus frequencies to a JSON file
        with open(corpus_freq_file, "w") as json_file:
            json.dump(corpus_freq, json_file, indent=2)


# Uni gram LM with Jelinek-Mercer smoothing
def run_jelinek():
    v = len(unique_terms)
    lambda_val = 0.9
    msearch_queries = []

    for query_text in queries.values():
        msearch_queries.append({"index": index_name})
        msearch_queries.append({"query": {"match": {"content": query_text}}, "_source": False, "size": 84678})

    # use the mulsti-search es api
    msearch_response = es.msearch(body=msearch_queries)

    for i, (query_number, query_text) in enumerate(queries.items()):
        output_lines = []
        hits = msearch_response["responses"][i].get("hits", {}).get("hits", [])
        for hit in hits:
            doc_id = hit["_id"]
            term_vectors = es.termvectors(index=index_name, id=doc_id, fields=["content"], term_statistics=True,
                                          positions=True)
            total_score = 0
            for query_term in query_text.split():
                term_info = term_vectors.get("term_vectors", {}).get("content", {}).get("terms", {}).get(query_term, {})
                tf = term_info.get("term_freq", -1000)
                first = lambda_val * (tf / len(doc_ids))
                second = corpus_freq.get(query_term) / v
                score = first + second
                if score > 0:
                    total_score += math.log(score)
                else:
                    total_score += -1000
            # Write results to the output file
            output_lines.append((total_score, doc_id))
        output_file = f"IR_data2/IR_data/AP_DATA/query_result_jelinek.txt"
        output_lines.sort(reverse=True, key=lambda x: x[0])
        valid_lines_written = 0
        with open(output_file, "a") as file:
            for rank, (score, doc_id) in enumerate(output_lines, start=1):
                if score != 0:
                    ranked_line = f"{query_number} Q0 {doc_id} {rank} {score} Exp\n"
                    file.write(ranked_line)
                    valid_lines_written += 1

                if valid_lines_written == 1000:
                    break

run_okapi_bm()
