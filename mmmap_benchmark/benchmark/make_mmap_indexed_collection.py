
from pprint import pprint
import time
from benchmark.settings import DATA_DIR

import os
from benchmark.upload_data import BenchmarkUpload


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", default='bench-01',
                        help="name of benchmark data file")

    args = parser.parse_args()
    bench_path = os.path.join(DATA_DIR, args.file_name)

    benchmark = BenchmarkUpload()
    benchmark.upload_data(path=bench_path, parallel=4)
    benchmark.wait_collection_green()
    benchmark.enable_mmap_and_index()

    time.sleep(0.5)

    pprint(benchmark.client.openapi_client.collections_api.get_collection(
        benchmark.collection_name).dict())

    wait_for_index_time = benchmark.wait_collection_green()
    print("Waited for index: ", wait_for_index_time)

    pprint(benchmark.client.openapi_client.collections_api.get_collection(
        benchmark.collection_name).dict())
