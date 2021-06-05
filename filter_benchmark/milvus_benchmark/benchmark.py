import json
import os
from collections import defaultdict
from pprint import pprint

import math
import numpy as np
from milvus import Milvus, DataType
from tqdm import tqdm

from sample_generator.settings import DATA_DIR


class MilvusBenchmark:

    def __init__(self, host='127.0.0.1', port='19530'):
        self.client = Milvus(host, port)
        self.collection_name = "filter_benchmark"

    def upload_data(self, path):
        vectors = np.load(path + '.npy')
        payloads = []

        vector_num, dim = vectors.shape
        print('vectors.shape', vectors.shape)

        with open(path + '.payload.jsonl', 'r') as fd:
            for line in fd:
                payloads.append(json.loads(line))

        collection_param = {
            "fields": [{
                "name": key,
                "type": DataType.INT32,
            } for key in payloads[0]] + [{
                "name": "embedding",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": dim}
            }],
            "auto_id": False
        }

        if self.collection_name in self.client.list_collections():
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(self.collection_name, collection_param)

        self.client.create_index(self.collection_name, 'embedding', {
            "index_type": "HNSW",
            "metric_type": "IP",  # one of L2, IP
            "params": {
                "M": 16,  # int. 4~64
                "efConstruction": 64  # int. 8~512
            }
        })

        ids = []
        
        embeddings = []
        for idx, vector in enumerate(vectors):
            ids.append(idx)
            embeddings.append(vector.tolist())

        print("Inserting .... ")
        batch_size = 10_000
        num_batches = math.ceil(len(embeddings) / batch_size)
        for batch_id in tqdm(range(num_batches)):
            
            payloads_batch = payloads[batch_id * batch_size:(batch_id + 1) * batch_size]
            embeddings_batch = embeddings[batch_id * batch_size:(batch_id + 1) * batch_size]
            ids_batch = ids[batch_id * batch_size:(batch_id + 1) * batch_size]
            
            entities = defaultdict(list)
            for payload_item in payloads_batch:
                for key, val in payload_item.items():
                    entities[key].append(val)

            hybrid_entities_batch = [
                              {
                                  "name": "embedding",
                                  "values": embeddings_batch,
                                  "type": DataType.FLOAT_VECTOR
                              }
                          ] + [
                              {
                                  "name": key,
                                  "values": vals,
                                  "type": DataType.INT32
                              } for key, vals in entities.items()
                          ]

            self.client.insert(self.collection_name, hybrid_entities_batch, ids_batch)


    def query(self, path):
        info = self.client.get_collection_info(self.collection_name)
        stats = self.client.get_collection_stats(self.collection_name)
        pprint(info)
        pprint(stats)
        queries = []
        with open(path + '.queries.jsonl', 'r') as fd:
            for line in fd:
                queries.append(json.loads(line))
        total_hits = 0
        expected_hits = 0
        for query in tqdm(queries):
            filter_conditions = [
                {
                    "term": {key: [val]}
                } for key, val in query['payload'].items()
            ]
            dsl = {
                "bool": {
                    "must": [
                        {
                            "vector": {
                                "embedding": {
                                    "topk": len(query['result']),
                                    "query": [query['vector']],
                                    "metric_type": "IP",
                                    "params": {
                                        "ef": 64
                                    },
                                }
                            }
                        }
                    ] + filter_conditions
                }
            }
            # pprint(dsl)

            result = self.client.search(self.collection_name, dsl=dsl)
            expected_ids = set(res for res, sim in query['result'])
            found_ids = set(result[0].ids)
            total_hits += len(found_ids.intersection(expected_ids))
            expected_hits += len(query['result'])
        print("hit_rate", total_hits / expected_hits)

