import os
import time
import json
from concurrent import futures
import httpx

from qdrant_client import QdrantClient

from benchmark.config import DATA_DIR


class BenchmarkSearch:

    def __init__(self, collection_name="benchmark_collection"):
        self.collection_name = collection_name
        self.client = QdrantClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=0))

    def search_one(self, query):
        true_result = set(res[0] for res in query['result'])
        top = len(true_result)
        start = time.time()
        res = self.client.search(
            self.collection_name,
            query_vector=query['vector'],
            top=top,
            append_payload=False
        )
        end = time.time()
        search_res = set(x.id for x in res)
        precision = len(search_res.intersection(true_result)) / top
        return precision, end - start

    def search_all(self, path, parallel_queries=4):
        queries = []
        with open(path + '.queries.jsonl', 'r') as fd:
            for line in fd:
                queries.append(json.loads(line))

        print(f"start querying with {parallel_queries} threads")
        start = time.time()
        with futures.ThreadPoolExecutor(max_workers=parallel_queries) as executor:
            future_results = executor.map(
                self.search_one,
                queries
            )

        precisions, latencies = list(zip(*future_results))

        end = time.time()

        print(f"avg precision = {sum(precisions) / len(precisions):.3f}")
        print(f"total time = {end - start:.3f} sec")
        print(f"time per query = {(end - start) / len(queries):.4f} sec")
        print(f"query latency = {sum(latencies) / len(queries):.4f} sec")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", default='bench-01',
                        help="name of benchmark data file")
    parser.add_argument("-p", "--parallel", default=4, type=int, help="number of threads for requests")

    args = parser.parse_args()
    bench_path = os.path.join(DATA_DIR, args.file_name)

    benchmark = BenchmarkSearch()
    benchmark.search_all(bench_path, parallel_queries=args.parallel)

