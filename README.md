
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

## Replication notes

1. Put OntoNotes 5.0 into `data/ontonotes-release-5.0`
2. Make sure you convert bracket files into dependency files (e.g. from 
`mnb_0001.parse` you would create `mnb_0001.dep`)
3. Run `python3 -m  hyperspace.prepare`
3. Run `python3 -m  hyperspace.adhoc_gen_data`
3. Run `python3 -m  hyperspace.exp_van_de_cruys_2014`
3. Run `python3 -m  hyperspace.exp_hyperspace`
