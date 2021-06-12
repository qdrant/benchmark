#!/bin/bash

docker run --rm -it --network=host --memory="${RAM_LIMIT}mb" -v "$(pwd)/data/storage:/qdrant/storage" -v "$(pwd)/config/config.yaml:/qdrant/config/production.yaml"  generall/qdrant

