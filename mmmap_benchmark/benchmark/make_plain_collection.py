
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
