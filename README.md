Source code for "Entity-for-mention Candidate Generation for Low-resource Languages"

# Data

Download the data from this [link](https://doi.org/10.5281/zenodo.3953649)

It contains 18 folders (one per language) which should be put inside the mentions_dumps folder.

The files `interlanguage_links_wikidata-20190901.csv` and `qid2title_CHAR.pkl` have to be in the same folder as the README file.

# Virtual Environment

Setup the virtual environment named `pti` to install all the required dependencies `conda env create -f pti.yml`.

Activate the installed environment `conda activate pti`

# Running PTI and Charagram

PTI and Charagram can be run by simply executing 

`python PTI/main.py alpha lambda target_lang pivot_lang`

and

`python CHARAGRAM/main.py mu amount_training_data target_lang pivot_lang`

respectively. Zero-shot setting is enabled by setting `alpha=-1` or `mu=-1`. Following guidelines by the authors of Charagram, `amount_training_data` is set to 80,000 for the main experiments contained in the paper. Both the pivot and target language are represented with the two or three character code indicated in the submission.

# Running WikiPriors

To run WikiPriors simply execute

`python WikiPriors/main.py args`.

where the arguments `args` are listed below:

- --tlang: target language.
- --plang: pivot language.
- --ncands: number of retrieved candidate entities. The default value is 30.
- --zeroshot: boolean to indicate whether the learning setting is zero-shot or not.
