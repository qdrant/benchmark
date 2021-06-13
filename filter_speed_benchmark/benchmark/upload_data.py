import json
import os
import time
from pprint import pprint

import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import Distance, CollectionStatus, StorageOperationsAnyOf1, \
    StorageOperationsAnyOf1UpdateCollection, OptimizersConfigDiff

from benchmark.config import DATA_DIR


class BenchmarkUpload:

    def __init__(self, collection_name="benchmark_collection"):
        self.collection_name = collection_name
        self.client = QdrantClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=0))

    def upload_data(self, path, parallel=4):

        vectors = np.load(path + '.npy')
        payloads = []

        vector_num, dim = vectors.shape
        print('vectors.shape', vectors.shape)

        with open(path + '.payload.jsonl', 'r') as fd:
            for line in fd:
                payloads.append(json.loads(line))

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vector_size=dim,
            distance=Distance.COSINE,
            optimizers_config=OptimizersConfigDiff(
                flush_interval_sec=10,
                indexing_threshold=1000000000,  # Disable indexing before all points are added
                memmap_threshold=1000000000,
                payload_indexing_threshold=1000000000,
                max_segment_number=4
            )
        )

        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors,
            payload=payloads,
            ids=None,
            parallel=parallel
        )

    def wait_collection_green(self):
        wait_time = 10.0
        total = 0
        collection_info = self.client.openapi_client.collections_api.get_collection(
            self.collection_name)
        while collection_info.result.status != CollectionStatus.GREEN:
            time.sleep(wait_time)
            total += wait_time
            collection_info = self.client.openapi_client.collections_api.get_collection(
                self.collection_name)
        return total

    def alter_config(self, diff: OptimizersConfigDiff):
        return self.client.openapi_client.collections_api.update_collections(
            StorageOperationsAnyOf1(update_collection=StorageOperationsAnyOf1UpdateCollection(
                name=self.collection_name,
                optimizers_config=diff
            ))
        )

    def set_indexed_field(self, field_name):
        self.client.create_payload_index(collection_name=self.collection_name, field_name=field_name)

    def enable_index(self):
        return self.alter_config(OptimizersConfigDiff(
            indexing_threshold=10000,
            payload_indexing_threshold=10000,
            max_segment_number=6
        ))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", default='bench-01', help="name of benchmark data file")
    parser.add_argument("-i", "--payload-index", default=None, help="enable index for a given payload")
    
    args = parser.parse_args()
    bench_path = os.path.join(DATA_DIR, args.file_name)

    benchmark = BenchmarkUpload()
    benchmark.upload_data(path=bench_path, parallel=4)
    if args.payload_index:
        print(f"Setting payload {args.payload_index}")
        benchmark.set_indexed_field(args.payload_index)
    
    benchmark.enable_index()
    benchmark.wait_collection_green()

    time.sleep(0.5)

    pprint(benchmark.client.openapi_client.collections_api.get_collection(
        benchmark.collection_name).dict())

    wait_for_index_time = benchmark.wait_collection_green()
    print("Waited for index: ", wait_for_index_time)

    pprint(benchmark.client.openapi_client.collections_api.get_collection(
        benchmark.collection_name).dict())
    
