
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
# Ensure that there is no data in page cache before each benchmark run
sudo bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches' 

# Run with docker memory limit
RAM_LIMIT=500 bash -x run-docker.sh
```

Upload search data

```bash
python -m benchmark.make_plain_collection # Upload and make in-memory collection
# or
python -m benchmark.make_plain_indexed_collection # Upload and make in-memory collection with HNSW index
# or
python -m benchmark.make_mmap_collection # Upload and make mmap-ed collection
# or
python -m benchmark.make_mmap_index_collection # Upload and make mmap-ed collection with HNSW index
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

* RAM Usage: 565Mb

```
total time = 14.746 sec
time per query = 0.0147 sec
query latency = 0.0588 sec
```

**Plain index mmap**

* Minimal RAM Usage: 243Mb
* Full page cache RAM usage: 464Mb

```
total time = 16.438 sec
time per query = 0.0164 sec
query latency = 0.0656 sec
```

**HNSW index with mmap**

Disk usage:

* Vectors size: 246M
* Vector index: 123M

RAM usage:

* Start RAM usage: 418Mb
* Full RAM without restrictions: 677Mb


|mem allowed, Mb|time per query|query latency|
|---------------|--------------|-------------|
|400            |0.1018        |0.4066       |
|450            |0.0991        |0.3958       |
|500            |0.0965        |0.3854       |
|550            |0.0558        |0.2228       |
|575            |0.0118        |0.0471       |
|600            |0.002         |0.008        |
|1000           |0.0019        |0.0077       |

