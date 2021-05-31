import os
import time
from pprint import pprint

import h5py
from qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import Distance, CollectionStatus, StorageOperationsAnyOf1, \
    StorageOperationsAnyOf1UpdateCollection, OptimizersConfigDiff

from search_benchmark.benchmark.config import DATA_DIR


class Benchmark:

    def __init__(self, collection_name="benchmark_collection"):
        self.collection_name = collection_name
        vectors_path = os.path.join(DATA_DIR, 'glove-100-angular.hdf5')
        self.data = h5py.File(vectors_path)
        self.client = QdrantClient()
        self.vector_size = len(self.data['train'][0])

    def upload_data(self, parallel=4):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
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
            vectors=self.data['train'],
            payload=None,
            ids=None,
            parallel=parallel
        )

    def wait_collection_green(self):
        wait_time = 2.0
        total = 0
        collection_info = self.client.openapi_client.collections_api.get_collection(self.collection_name)
        while collection_info.result.status != CollectionStatus.GREEN:
            time.sleep(wait_time)
            total += wait_time
            collection_info = self.client.openapi_client.collections_api.get_collection(self.collection_name)
        return total

    def enable_indexing(self):

        self.client.openapi_client.collections_api.update_collections(
            StorageOperationsAnyOf1(update_collection=StorageOperationsAnyOf1UpdateCollection(
                name=self.collection_name,
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            ))
        )


if __name__ == '__main__':

    benchmark = Benchmark()
    benchmark.upload_data(parallel=4)
    benchmark.wait_collection_green()
    benchmark.enable_indexing()

    time.sleep(0.5)

    # pprint(benchmark.client.openapi_client.collections_api.get_collection(benchmark.collection_name).dict())
    #
    # wait_for_index_time = benchmark.wait_collection_green()
    # print("Waited for index: ", wait_for_index_time)
    #
    # pprint(benchmark.client.openapi_client.collections_api.get_collection(benchmark.collection_name).dict())

