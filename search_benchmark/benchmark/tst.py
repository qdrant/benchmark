import os
import time
from concurrent import futures
import requests
import httpx

import h5py
from qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import SearchRequest

from benchmark.config import DATA_DIR


if __name__ == '__main__':

    vectors_path = os.path.join(DATA_DIR, 'glove-100-angular.hdf5')
    data = h5py.File(vectors_path)
    

    collection_name = "benchmark_collection"

    client = QdrantClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=0))

    for i in range(150):

        start = time.time()

        vector = list(data['test'][i])

        res = client.http.points_api.search_points(
            name=collection_name,
            search_request=SearchRequest(
                vector=vector,
                filter=None,
                top=10,
                params=None
            )
        )

        # requests.post(f"http://localhost:6333/collections/{collection_name}/points/search", json={
        #     "vector": list(map(int, vector)),
        #     "top": 10
        # })


        end = time.time()

        print((end - start) * 1000)
        # print(f"{res.time * 1000:.2f}, {(end - start) * 1000:.2f}")