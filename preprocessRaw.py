#! python

import os
import pickle
import time
from sys import argv, exit

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tag.perceptron import PerceptronTagger

"""
Apply POS tagging to raw text.
Example:
The sentence
    The Cat only grinned when it saw Alice.
turns into
    The_DT Cat_NNP only_RB grinned_VBD when_WRB it_PRP saw_VBD Alice_NNP ._.
"""

if __name__ == "__main__":
    # Verify argument
    if len(argv) < 2:
        print("Requires .txt file as argument")
        exit()

    # Set IO paths
    DIR     = os.getcwd()
    INPATH  = DIR + "/raw/" + argv[1] + ".txt"
    OUTPATH = DIR + "/preproc/" + argv[1] + ".preproc"
    # Read and preprocess text, line by line
    tagger = PerceptronTagger()
    preproc = ''
    start = time.time()
    with open(INPATH,'r') as f:
        for line in f:
            line = line.replace('[','')
            line = line.replace(']','')
            line = line.replace('*','')
            line = line.replace('_','')
            line = line.replace('<s>','')
            line = line.replace('</s>','.')
            preproc += ' '.join(token+'_'+tag for token,tag in tagger.tag(word_tokenize(line.lower()))) + ' '
    print(time.time()-start)

    # Write to file
    with open(OUTPATH,'w') as f:
        f.write(preproc)
    print("finished writing to file")