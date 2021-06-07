
# Qdrant search speed benchmark


## Data

Based on https://github.com/erikbern/ann-benchmarks#data-sets

GloVe Angular 100: http://ann-benchmarks.com/glove-100-angular.hdf5


## Usage


Download data

```bash
bash get_data.sh
```

Install dependencies

```bash
pip install -r requirement.txt
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

For GloVe Angular 100 dataset: 

* `num_vectors = 1,183,514`
* `dim = 100`
* `metric = cosine`

Used HNSW index params:

* `M = 16`
* `efConstruct = 100`
* `ef = 100`

```
Search with 1 threads
avg precision = 0.876
total time = 94.418 sec
time per query = 0.0094 sec
query latency = 0.0034 sec
```


```
Search with 4 threads
avg precision = 0.876
total time = 28.030 sec
time per query = 0.0028 sec
query latency = 0.0087 sec
```
