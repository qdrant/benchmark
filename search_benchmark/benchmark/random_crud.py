import json
import random
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models.models import PointRequest, PointsList, PointStruct, PointIdsList

import httpx
import tqdm


class CrudTester:

    def __init__(self, schema, num_vectors, dim):
        self.dim = dim
        self.num_vectors = num_vectors
        self.schema = schema
        self.client = QdrantClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=0))

        self.timings = {
            "delete": [],
            "update": [],
            "get": []
        }

    def random_vector(self):
        return [
            random.random()
            for _ in range(self.dim)
        ]

    def random_payload(self):
        return {
            # "int_payload": random.randint(0, 1_000_000),
            # "array_payload": [
            #     random.randint(0, 1_000_000),
            #     random.randint(0, 1_000_000),
            #     random.randint(0, 1_000_000),
            # ],
            # "string_payload": uuid.uuid4().hex
        }

    def random_id(self):
        return random.randint(0, self.num_vectors)

    def delete(self):
        idx = self.random_id()
        start_time = time.monotonic()
        self.client.http.points_api.delete_points(
            collection_name=self.schema,
            wait=True,
            points_selector=PointIdsList(points=[idx])
        )
        elapsed = time.monotonic() - start_time
        self.timings["delete"].append(elapsed)

    def update(self):
        start_time = time.monotonic()
        self.client.http.points_api.upsert_points(
            collection_name=self.schema,
            wait=True,
            point_insert_operations=PointsList(
                points=[
                    PointStruct(id=self.random_id(), payload=self.random_payload(), vector=self.random_vector())
                ]
            )
        )

        elapsed = time.monotonic() - start_time
        self.timings["update"].append(elapsed)

    def get(self):
        start_time = time.monotonic()
        self.client.http.points_api.get_points(
            self.schema,
            point_request=PointRequest(ids=[self.random_id()])
        )
        elapsed = time.monotonic() - start_time
        self.timings["get"].append(elapsed)

    def burn(self, count: int = 1000):
        with tqdm.tqdm(total=count) as pbar:
            total = 0
            while total < count:
                new_total = len(self.timings['delete']) + len(self.timings['update']) + len(self.timings['get'])
                diff = new_total - total
                if diff:
                    pbar.update(n=diff)

                total = new_total
                action = random.randint(0, 3)
                if action == 0:
                    self.get()
                if action == 1:
                    self.update()
                if action == 2:
                    self.delete()

    def save_timings(self, path):
        with open(path, 'w') as out:
            json.dump(self.timings, out)


if __name__ == '__main__':
    tester = CrudTester('benchmark_collection', num_vectors=1_000_000, dim=100)

    tester.burn(count=10000)

    print("Avg delete time", sum(tester.timings['delete']) / len(tester.timings['delete']))
    print("Avg update time", sum(tester.timings['update']) / len(tester.timings['update']))
    print("Avg get time", sum(tester.timings['get']) / len(tester.timings['get']))

    tester.save_timings("crud_timings.json")
