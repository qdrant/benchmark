import os
import time
from concurrent import futures

import h5py
from qdrant_client import QdrantClient

from benchmark.config import DATA_DIR


class BenchmarkSearch:

    def __init__(self, collection_name="benchmark_collection"):
        self.collection_name = collection_name
        vectors_path = os.path.join(DATA_DIR, 'glove-100-angular.hdf5')
        self.data = h5py.File(vectors_path)
        self.client = QdrantClient()
        self.vector_size = len(self.data['test'][0])

    def search_one(self, i):
        top = 10
        true_result = set(self.data['neighbors'][i][:top])

        res = self.client.search(
            self.collection_name,
            query_vector=self.data['test'][i],
            top=top,
            append_payload=False
        )
        search_res = set(x.id for x in res)
        precision = len(search_res.intersection(true_result)) / top
        return precision

    def search_all(self, parallel_queries=4):
        num_queries = len(self.data['test'])
        start = time.time()

        with futures.ThreadPoolExecutor(max_workers=parallel_queries) as executor:
            future_results = executor.map(
                self.search_one,
                range(num_queries)
            )

        precisions = list(future_results)

        end = time.time()

        print(f"avg precision = {sum(precisions) / len(precisions)}")
        print(f"total time = {end - start}")
        print(f"time per query = {(end - start) / num_queries}")


if __name__ == '__main__':
    benchmark = BenchmarkSearch()
    benchmark.search_all()
