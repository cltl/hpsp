
Initial experiments: comparing hyperspace model and van de Cruys (2014).

When evaluated on a randomly generated pseudo-disambiguation dataset
(`output/dep_context_labeled.dev.txt`), the model
of van de Cruys (2014) gets incredibly high result: 94.08%.

When controlled for frequency (`output/dep_context_labeled.dev-fixed.txt`),
the same model gives only 51.81%, i.e. barely better than chance.

This is a big problem: all results before and including van de Cruys (2014)
need to be reconsidered. A better evaluation set needs to be constructed. 

My model gives slightly higher accuracy (once 53.96% and another time 54.37%)
but it also gives 0% for random seed 19. I guess it got NaN or inf in that case
but I'll need to check again.

## Dependencies

TensorFlow 1.0+

## Data

- Download OntoNotes
- Use [this script](https://bitbucket.org/cltl/isrl-sp/src/756fadf8d1d25d6a4271f0cc4caa94af0ab095da/constituency2dependency.py?at=master&fileviewer=file-view-default) to create dependency trees

## Replication notes

3. `git checkout d63ddf1` and run the notebook `notebook/extracting-triples.ipynb`
4. 