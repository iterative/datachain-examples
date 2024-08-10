## What the demo shows

* There is a container on Azure with all neurips conference papers as well as some bibtex metadata.
* We use DVCx and its UDFs to read all of these PDFs, clean them up, apply embedding and add them to a structured format
* The results are versioned datasets which are essentially published data products, ready for downstream consumption
* It would also be possible to process scanned documents (e.g. contracts) in the same way, only using a different UDF for the initial processing.

## How to run

Use `demo.ipynb` when running locally. In the Studio UI, use `demo-studio.py` as a script instead and upload `text_loaders.py` as an attached file. Fill in requirements from `requirements.txt`

## Comments
* The neurips data are on Azure in the container `https://storageubs.blob.core.windows.net/neurips`. It does not contain the entire dataset available [here](https://papers.nips.cc/) but it is sufficiently big for demonstrations (and faster to process).
* In Studio you need to uncomment the `source="az://neurips"` line in the notebook to use the right source.
* When running locally, it might be better to use the `neurpis` folder versioned by DVC which only contains data from 2 years and so it can be used for faster iterations.
* It is possible to combine the PDF data with corresponding bibtex data, but this requires more code and in the end doesn't seem to show much more than the ability to use join operations on DataChain, omitted for simplicity
