# Style datasets

This directory contains all scripts for style dataset preprocessing.
Additionally, it creates the train, validation and test splits in the `data` directory for each style corpus.

## Datasets

The preprocessing steps for the four corpora are in the `preprocess-*.py` scripts.
We briefly describe the datasets in the following.

### GYAFC

We use the same splits of the [Grammarly's Yahoo Answers Formality Corpus](https://github.com/raosudha89/GYAFC-corpus) (GYAFC) as STRAP. Refer to the [instructions](https://github.com/martiansideofthemoon/style-transfer-paraphrase#datasets) of STRAP to get access to the formality data. Place the data in the `data/gyafc/raw/` directory.

### OLID

We use the [Offensive Language Identification Dataset](https://scholar.harvard.edu/malmasi/olid) (OLID) for offensiveness style transfer. Follow the download instructions and place the data in the `data/olid/raw` directory.

### Politeness

For politeness style transfer, we use the automatically labelled [dataset](https://github.com/tag-and-generate/politeness-dataset) by Madaan et al. Download the dataset and place it in the `data/politeness/raw` directory.


### Receptiveness

We use publicly available [data](https://osf.io/2n59b/?view_only=48308aeeb44d4b5eafeeaf476f224527) from Mike Yeomans as well as some unreleased conversations from the same author for receptiveness style transfer.
Please contact us to get access to this data.
