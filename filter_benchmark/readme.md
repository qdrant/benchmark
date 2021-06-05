# Filtrable search benchmark

Testing the completeness of search results when using filters.
Using filters in conjunction with the HNSW graph without any additional measures leads to loss of results due to loss of graph connectivity.
More information here: https://blog.vasnetsov.com/posts/categorical-hnsw/


## Example Usage


Install dependencies
```
pip install poetry
poetry install
```

Create sample data collection (numpy).
10 million vectors with dim=16 (default) + payload of 200 different values.

```
python -m sample_generator.generator -n 10000000 -p 200 -q 100
```

Run Milvus

```
docker-compose up
```

Upload data into milvus
```
python -m milvus_benchmark.upload_data
```

Search for test queries with filter
```
python -m milvus_benchmark.search
```

Expected output would include collection information + estimates precision@10 for test queries.

Result for given params:
```
hit_rate 0.778
```

That would mean that only 77.8% of results were found, compared to exact search.

Note, that search without filters give almost perfect result with given parameters of HNSW index.

