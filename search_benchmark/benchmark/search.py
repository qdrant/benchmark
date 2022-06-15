import json
import os
import time
from multiprocessing import Pool

import h5py
import httpx
from qdrant_client import QdrantClient

from benchmark.config import DATA_DIR


class Querier:
    client = None
    collection_name = None

    @classmethod
    def init_client(cls, collection_name="benchmark_collection"):
        cls.collection_name = collection_name
        cls.client = QdrantClient(
            prefer_grpc=True,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=0),
        )

    @classmethod
    def search_one(cls, params):
        neighbors, vector = params
        top = 10
        true_result = set(neighbors[:top])
        start = time.monotonic()
        res = cls.client.search(
            collection_name=cls.collection_name,
            query_vector=vector,
            top=top,
            with_payload=False
        )
        end = time.monotonic()
        search_res = set(x.id for x in res)
        precision = len(search_res.intersection(true_result)) / top
        return precision, end - start


class BenchmarkSearch:

    def __init__(self, data, collection_name="benchmark_collection"):
        self.collection_name = collection_name
        self.data = data
        self.client = QdrantClient(
            prefer_grpc=True,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=0)
        )
        self.vector_size = len(self.data['test'][0])

    def search_one(self, i):
        top = 10
        true_result = set(self.data['neighbors'][i][:top])
        start = time.time()
        res = self.client.search(
            self.collection_name,
            query_vector=self.data['test'][i],
            top=top,
            with_payload=False
        )
        end = time.time()
        search_res = set(x.id for x in res)
        precision = len(search_res.intersection(true_result)) / top
        return precision, end - start

    def search_all(self, parallel_queries=4):
        num_queries = len(self.data['test'])
        print(f"Search with {parallel_queries} threads")
        start = time.time()

        future_results = []
        if parallel_queries == 1:
            for i in range(num_queries):
                res = self.search_one(i)
                future_results.append(res)
            precisions, latencies = list(zip(*future_results))
        else:
            with Pool(processes=parallel_queries, initializer=Querier.init_client, initargs=(self.collection_name,)) as pool:
                precisions, latencies = list(zip(*pool.imap_unordered(
                    Querier.search_one,
                    iterable=zip(self.data['neighbors'], self.data['test']))))

        end = time.time()

        print(f"avg precision = {sum(precisions) / len(precisions):.3f}")
        print(f"total time = {end - start:.3f} sec")
        print(f"time per query = {(end - start) / num_queries:.4f} sec")
        print(f"query latency = {sum(latencies) / len(latencies):.4f} sec")

        return latencies


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallel", default=4, type=int, help="number of threads for requests")

    args = parser.parse_args()

    vectors_path = os.path.join(DATA_DIR, 'glove-100-angular.hdf5')
    data = h5py.File(vectors_path)

    benchmark = BenchmarkSearch(data)
    latencies = benchmark.search_all(parallel_queries=args.parallel)

    json.dump({"latencies": latencies}, open("search_latencies.json", 'w'))
