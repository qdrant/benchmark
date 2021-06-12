
# Qdrant memory benchmark

- Estimate disk usage
- Estimate memory consumption plain vs mmap
- Estimate search speed plain vs mmap


## Data

Randomly generated vectors

## Usage


Install dependencies

```bash
pip install poetry
poetry install
```

Generate data

```bash
python -m benchmark.generate_data -n 1000000 -d 64 -q 1000
```

Run Qdrant instance with 500 Mb ram limit:

```bash
sudo bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches' # Ensure that there is no data in page cache before each benchmark run

RAM_LIMIT=500 bash -x run-docker.sh
```

Upload search data

```bash
python -m benchmark.upload_data
```
Uploaded data available for search immediately, but building of HNSW index may take quite some time. 

Run search benchmark

```bash
python -m benchmark.search
```


## Example Results

Qdrant params:

* Num parallel searchers: 4
* Num parallel queries: 4

For GloVe Angular 100 dataset: 

* `num_vectors = 1.000.000`
* `dim = 64`
* `metric = cosine`

Used HNSW index params:

* `M = 16`
* `efConstruct = 100`
* `ef = 100`


**Plain index in-memory**

```
total time = 14.746 sec
time per query = 0.0147 sec
query latency = 0.0588 sec
```

<!-- RAM usage: 565Mb -->

**Plain index mmap**

* Start RAM Usage: 243Mb

```
total time = 16.438 sec
time per query = 0.0164 sec
query latency = 0.0656 sec
```

<!-- Mem usage: 243Mb, after search - 464Mb -->

**HNSW index with mmap**

* Start RAM usage: 418Mb
* Full RAM usage without mmap: 677Mb


Benchmark with RAM Limit 450Mb (12% full vector size)

```
total time = 99.117 sec
time per query = 0.0991 sec
query latency = 0.3958 sec
```

Benchmark with RAM Limit 500Mb (31% full vector size)

```
total time = 96.518 sec
time per query = 0.0965 sec
query latency = 0.3854 sec
```

Benchmark with RAM Limit 550Mb (50% full vector size)

```
total time = 55.811 sec
time per query = 0.0558 sec
query latency = 0.2228 sec
```

Benchmark with RAM limit 600Mb (71% full vector size)

```
total time = 1.932 sec
time per query = 0.0019 sec
query latency = 0.0077 sec
```
