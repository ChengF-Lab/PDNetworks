# A network-based systems genetics framework identifies pathobiology and drug repurposing in Parkinson’s disease

## Requirements

See [environment.yml](environment.yml)

## Installation

- `conda env create --file environment.yml`

## Usage

1. Prepare xQTL-regulated data as input (see [data](data folder)):

2. Risk gene prediction

```bash
bash ./code/run.sh
##
Default parameters in the run.sh and main.py file are the ones used in the manuscript.
In order to improve prediction stability, we ensemble results with 10 different seed.
For the sake of convenience, users can provide your own directories of functional genomics data (e.g., regulatory element or xQTL) into config file (see our example config file 'data/qtl.conf' in data folder).

Validation of predicted pdRGs via snRNA-seq data can be see here (./snRNA-seq_validation)
