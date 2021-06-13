import os
import time
import json
from concurrent import futures
import httpx

from qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import FieldCondition, Filter, Match

from benchmark.config import DATA_DIR


class BenchmarkSearch:

    def __init__(self, collection_name="benchmark_collection"):
        self.collection_name = collection_name
        self.client = QdrantClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=0))

    def search_one(self, query):
        true_result = set(res[0] for res in query['result'])
        top = len(true_result)
        if query['payload']:
            must_clauses = [
                FieldCondition(key=key, match=Match(integer=value)) 
                for key, value in query['payload'].items()
            ]
            query_filter = Filter(must=must_clauses)
        else:
            query_filter = None

        start = time.time()
        res = self.client.search(
            self.collection_name,
            query_vector=query['vector'],
            query_filter=query_filter,
            top=top,
            append_payload=False
        )
        end = time.time()
        search_res = set(x.id for x in res)
        precision = len(search_res.intersection(true_result)) / top
        return precision, end - start

    def search_all(self, path, query_field=None, parallel_queries=4):
        queries_path = path + (f'.queries-{query_field}.jsonl' if query_field else f'.queries.jsonl')

        queries = []
        with open(queries_path, 'r') as fd:
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
    parser.add_argument("-q", "--query", default=None, help="query with payload")

    args = parser.parse_args()
    bench_path = os.path.join(DATA_DIR, args.file_name)

    benchmark = BenchmarkSearch()
    benchmark.search_all(bench_path, query_field=args.query, parallel_queries=args.parallel)

