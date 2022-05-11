import os
import time

import h5py
from qdrant_client import QdrantClient
from qdrant_client.http.models.models import Distance, CollectionStatus, OptimizersConfigDiff, UpdateCollection

from benchmark.config import DATA_DIR


class Benchmark:

    def __init__(self, data, collection_name="benchmark_collection"):
        self.collection_name = collection_name
        self.data = data
        self.client = QdrantClient(prefer_grpc=True)
        self.vector_size = len(self.data['train'][0])

    def upload_data(self, parallel=4):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            distance=Distance.COSINE,
            optimizers_config=OptimizersConfigDiff(
                flush_interval_sec=10,
                indexing_threshold=10000000,  # For better speed: Disable indexing before all points are added
                memmap_threshold=1000000000,
                payload_indexing_threshold=1000000000,
            )
        )

        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=self.data['train'],
            payload=None,
            ids=None,
            parallel=parallel
        )

    def wait_collection_green(self):
        wait_time = 1.0
        total = 0
        collection_info = self.client.openapi_client.collections_api.get_collection(self.collection_name)
        while collection_info.result.status != CollectionStatus.GREEN:
            time.sleep(wait_time)
            total += wait_time
            collection_info = self.client.openapi_client.collections_api.get_collection(self.collection_name)
        return total

    def enable_indexing(self):
        self.client.http.collections_api.update_collection(
            collection_name=self.collection_name,
            update_collection=UpdateCollection(
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000,
                )
            )
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallel", default=4, type=int, help="number of threads for requests")

    args = parser.parse_args()

    vectors_path = os.path.join(DATA_DIR, 'glove-100-angular.hdf5')
    data = h5py.File(vectors_path)

    start = time.monotonic()
    benchmark = Benchmark(data)
    benchmark.upload_data(parallel=args.parallel)
    benchmark.wait_collection_green()
    end_upload = time.monotonic()

    print("Data uploaded: ", end_upload - start)
    benchmark.enable_indexing()
    time.sleep(0.5)
    # pprint(benchmark.client.openapi_client.collections_api.get_collection(benchmark.collection_name).dict())

    wait_for_index_time = benchmark.wait_collection_green()
    end_indexing = time.monotonic()
    print("Index created: ", end_indexing - start)

    # pprint(benchmark.client.openapi_client.collections_api.get_collection(benchmark.collection_name).dict())

