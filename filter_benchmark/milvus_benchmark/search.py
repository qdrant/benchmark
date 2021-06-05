from milvus_benchmark.benchmark import MilvusBenchmark
import os
from sample_generator.settings import DATA_DIR


if __name__ == '__main__':
    import string
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file-name", default='bench-01', help="name of benchmark data file")

    args = parser.parse_args()

    bench_path = os.path.join(DATA_DIR, args.file_name)
    benchmark = MilvusBenchmark()
    benchmark.query(bench_path)
