Experiments with normal activation function (e^(-x^2)).

Constructed an evaluation set from OntoNotes: 10k pairs of good and bad s-v-o tuples.
The corrupted tuples are created by selecting random subjects and (direct) objects
in the same frequency band as the good one. The control for frequency makes it much 
harder than previous datasets. TODO: validate this new dataset with human annotators.

TODO: print out words that activate each neuron the most. Compare activation functions
in terms of what property they capture. Also compare the number of words that activate
a neuron.

TODO: study the effect on word embeddings -- would we able to capture more
similarity instead of relatedness?

## Data

1. Download OntoNotes
2. Use [this script](https://bitbucket.org/cltl/isrl-sp/src/756fadf8d1d25d6a4271f0cc4caa94af0ab095da/constituency2dependency.py?at=master&fileviewer=file-view-default) to create dependency trees
3. Download dependency-parsed [ukWaC](http://wacky.sslmit.unibo.it/doku.php?id=corpora) 
and put under `/data/ukwac-dep`

## Replication

1. Install dependencies: see `requirements.txt`
2. Install [drake](https://github.com/Factual/drake) 
3. Run from the command line: `drake`
4. Results are in `notebook/*.html`
    - `cruys2014.html`: evaluating Cruys (2014) model