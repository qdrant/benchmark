
# Qdrant memory benchmark

- Estimate speed with and without filters


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
python -m benchmark.generate_data -n 1000000 -d 64 -q 1000 -p 5
```

Run Qdrant instance:

```bash
docker-compose up
```

Upload search data and build index for field `a`

```bash
python -m benchmark.upload_data -i a
```

Uploaded data available for search immediately, but building of HNSW index may take quite some time. 

Run search benchmark

```bash
# Without filters
python -m benchmark.search
# With filters
python -m benchmark.search -q a
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


**Speed without filters**

```
total time = 2.240 sec
time per query = 0.0022 sec
query latency = 0.0089 sec
```

**Speed with filter**

```
total time = 4.666 sec
time per query = 0.0047 sec
query latency = 0.0184 sec
```
