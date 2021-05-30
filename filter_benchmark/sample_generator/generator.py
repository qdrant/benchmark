import json
import os
import random
from pprint import pprint

from scipy.spatial import distance
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from sample_generator.settings import DATA_DIR


class SampleGenerator:
    @classmethod
    def generate_data(cls, num, dim, payload_params: dict):
        payloads = []
        vectors = normalize(np.random.rand(num, dim).astype(np.float32))

        for _ in range(num):
            payload = {}
            for key, max_val in payload_params.items():
                payload[key] = random.randint(0, max_val)
            payloads.append(payload)

        return vectors, payloads

    @classmethod
    def check_payload(cls, stored: dict, query: dict):
        for key, val in query.items():
            if stored.get(key) != val:
                return False
        return True

    def __init__(self, num, dim, payload_params: dict):
        self.dim = dim
        self.num = num
        self.payload_params = payload_params
        self.vectors, self.payloads = self.generate_data(num, dim, payload_params)

    def search(self, vector, payload, top=10):
        mask = np.array(list(map(lambda x: self.check_payload(x, payload), self.payloads)))
        # Select only matched by payload vectors
        filtered_vectors = self.vectors[mask]
        # List of original ids
        raw_ids = np.arange(0, self.num)
        # List of ids, filtered by payload
        filtered_ids = raw_ids[mask]
        # Scores among filtered vectors
        scores = cosine_similarity([vector], filtered_vectors)[0]
        # Ids in filtered matrix
        top_scores_ids = np.argsort(scores)[-top:][::-1]
        top_scores = scores[top_scores_ids]
        # Original ids before filtering
        original_ids = filtered_ids[top_scores_ids]
        return list(zip(map(int, original_ids), map(float, top_scores)))

    def generate_query(self, payload_params=None) -> dict:
        if payload_params is None:
            payload_params = self.payload_params
        vector_query = normalize(np.random.rand(self.dim).astype(np.float32).reshape(1, -1))[0]
        payload = {}

        for key in payload_params:
            max_val = self.payload_params[key]
            payload[key] = random.randint(0, max_val)

        return {
            "vector": list(map(float, vector_query)),
            "payload": payload,
            "result": self.search(vector_query, payload)
        }

    def generate(self, path, num_queries, payload_keys=None):
        np.save(path + '.npy', self.vectors)
        with open(path + '.payload.jsonl', 'w') as out:
            for payload in self.payloads:
                out.write(json.dumps(payload))
                out.write('\n')

        with open(path + '.queries.jsonl', 'w') as out:
            for _ in tqdm(range(num_queries)):
                out.write(json.dumps(self.generate_query(payload_keys)))
                out.write('\n')


if __name__ == '__main__':
    num_vectors = 1_000_000
    dim = 4
    payload_params = {'a': 2, 'b': 4, 'c': 8, 'e': 16, 'f': 32, 'g': 64}
    sampler = SampleGenerator(num_vectors, dim, payload_params=payload_params)
    sampler.generate(os.path.join(DATA_DIR, "bench-02"), num_queries=100, payload_keys={'g'})
