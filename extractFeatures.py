#! python

import os
import pickle
from sys import argv, exit

from collections import Counter

from nltk import ngrams


"""
Extract features from instance of preposition occurence within a POS-tagged text
Example:
"""

def getLexicalFromNgram(n_gram):
# Extract lexical info from the n_gram
    lexical = '_'.join(i.split('_')[0] for i in n_gram)
    return lexical

def getPOSFromNgram(n_gram):
# Extract POS info from the n_gram
    pos = '_'.join(i.split('_')[1] for i in n_gram)
    return pos

if __name__ == "__main__":
    # Verify argument
    if len(argv) < 2:
        print("""Requires .preproc file as argument""")
        exit()

    # Set IO paths
    DIR     = os.getcwd()
    INPATH  = DIR + "/preproc/"+argv[1]+".preproc"
    OUTPATH = DIR + "/features/"+argv[1]+".features.pickle"

    # Ngram window size, both ways
    window  = 9
    mid     = int(window/2)

    # Number of prepositions (most common) for training
    prep_no = 15
    
    # Load preprocessed text
    with open(INPATH, 'r') as f:
        preproc = f.read()

    # Create list of preposition samples
    samples = []

    # Iterate through the n-grams generated from preproc
    for n_gram in ngrams(preproc.split(),window):
        # Look at the token at the centre of the n_gram
        pair            = n_gram[mid]
        (token, tag)    = pair.split("_")

        # Detect if said token is a preposition
        # (POS tag = TO for 'to', IN for the rest)
        if tag in ('IN','TO'):
            # If infinitive marker 'to', ignore token
            if tag == 'TO' and n_gram[mid+1].split('_')[1] == 'VB' :
                continue

            # Get features
            samples.append(
                dict(
                    label=token,
                    # Lexical ngrams
                    # Before token
                    Lexical4B   = getLexicalFromNgram(n_gram[mid-4:mid]),
                    Lexical3B   = getLexicalFromNgram(n_gram[mid-3:mid]),
                    Lexical2B   = getLexicalFromNgram(n_gram[mid-2:mid]),
                    Lexical1B   = getLexicalFromNgram(n_gram[mid-1:mid]),
                    # After token
                    Lexical1A   = getLexicalFromNgram(n_gram[mid+1:mid+2]),
                    Lexical2A   = getLexicalFromNgram(n_gram[mid+1:mid+3]),
                    Lexical3A   = getLexicalFromNgram(n_gram[mid+1:mid+4]),
                    Lexical4A   = getLexicalFromNgram(n_gram[mid+1:mid+5]),

                    # POS ngrams
                    # Before token
                    POS4B       = getPOSFromNgram(n_gram[mid-4:mid]),
                    POS3B       = getPOSFromNgram(n_gram[mid-3:mid]),
                    POS2B       = getPOSFromNgram(n_gram[mid-2:mid]),
                    POS1B       = getPOSFromNgram(n_gram[mid-1:mid]),
                    # After token
                    POS1A       = getPOSFromNgram(n_gram[mid+1:mid+2]),
                    POS2A       = getPOSFromNgram(n_gram[mid+1:mid+3]),
                    POS3A       = getPOSFromNgram(n_gram[mid+1:mid+4]),
                    POS4A       = getPOSFromNgram(n_gram[mid+1:mid+5]),
                )
            )
        
    # Removing outliers (Wrong POS [nltk.pos_tag isn't perfect], uncommon instances)
    # Count frequency of each label
    label_counts = Counter(i['label'] for i in samples)
    # Select most common labels
    common_labels = []
    for label,_ in label_counts.most_common(prep_no):
        common_labels.append(label)
    # Select samples of most common labels
    common_samples = []
    for s in samples:
        if s['label'] in common_labels:
            common_samples.append(s)

    # Saving samples
    with open(OUTPATH, 'wb') as f:
        pickle.dump(common_samples,f)
