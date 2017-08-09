# UROP 2017: Grammatical Error Correction - Preposition Classifier
Khanh Nguyen, August 2017

A classifier to select the most suitable preposition for a given context, trained from raw text.

## Directory
* preprocessRaw.py: preprocess (pos tag) raw text
* extractFeatures.py: extract preposition instances and features from preprocessed file
* trainClassifier.py: vectorize features, train and evaluate classifier
* ./raw: contains 18 books from the gutenberg project
* ./preproc: contains preprocessed files
* ./features: contains pickle files of instances extracted from preprocessed files

## Python libraries:
python 3
os
sys
time
pickle
collections
nltk
sklearn

## Quickstart:
A quick example to get started. Let us train a classifier from the the text of Moby Dick (./raw/melville-moby_dick).

On the command line/terminal, go to this directory (./GEC_prep_classifier) and enter the following commands.

### 1. Preprocess the text
> python preprocessRaw.py melville-moby_dick

It may take a while for the text to finish preprocessing. The preprocessed file is written to ./preproc
[**Altenatively:** the [Stanford tagger](https://nlp.stanford.edu/software/tagger.shtml) accomplishes the same task, with the same output format and everything. And it is quicker too. It doesn't have the memory for extremely large .txt files, however. ie. ~1 GB]

### 2. Extract features
> python extractFeatures.py melville-moby_dick

The file is written as a pickle file to ./features

### 3. Train classifier
> python trainClassifier.py melville-moby_dick

