
# Qdrant mmap benchmark

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

Run Qdrant instance

```bash
docker-compose up
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
avg precision = 1.000
total time = 15.061 sec
time per query = 0.0151 sec
query latency = 0.0601 sec
```

**HNSW index in-memory**

```
avg precision = 0.991
total time = 4.538 sec
time per query = 0.0045 sec
query latency = 0.0180 sec
```

**Plain index mmap**

```
avg precision = 1.000
total time = 2.701 sec
time per query = 0.0027 sec
query latency = 0.0107 sec
```

**HNSW index with mmap**


```
avg precision = 1.000
total time = 77.123 sec
time per query = 0.0771 sec
query latency = 0.3080 sec
```
