# Accidents In Rome

## Requirements

Datasets are needed to run the notebook. You can either download them from Roma Open Data's site and then run the preprocessing script. Otherwise download a pre-processed ready for use version from the Releases page of this repository.

For running the scripts and the notebook you need a Python 3.9+ installation with Pandas and a container runtime like Docker or Podman.

## Preprocessing data

Since many rows in the accidents don't have a coordinate but just a street name, we need to obtain the coordinates ("geocoding") by running a Nominatim instance. To do so please run the following:

```bash
# Run Nominatim
./scripts/run-nominatim.sh

# Do the pre-processing
./scripts/preprocessing.py datasets/raw
```

## Running the notebook

From the root directory:

```bash
./scripts/run-jupyter.sh
```

## References

- [Accidents datasets](https://dati.comune.roma.it/catalog/dataset?q=incidenti&sort=score+desc%2C+dataset_last_update+desc)
