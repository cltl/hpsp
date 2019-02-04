import numpy as np
import tensorflow as tf


class DynamicDataset(object):
    
    def __init__(self, positive_examples):
        # positive_examples = positive_examples[:10000] # debugging
        self.positive_examples = positive_examples
        self.verb = positive_examples[:,[0]]
        self.pos_sbj = positive_examples[:,[1]]
        self.pos_dobj = positive_examples[:,[2]]
        self.neg_sbj = self.pos_sbj.copy()
        self.neg_dobj = self.pos_dobj.copy()
        self.y = np.ones((len(positive_examples), 1)) # dummy scores
        self.rng = np.random.RandomState(3892)
        
    def generate(self):
        self.rng.shuffle(self.neg_sbj)
        self.rng.shuffle(self.neg_dobj)
        input_pos = self.positive_examples
        input_neg1 = np.hstack([self.verb, self.neg_sbj, self.pos_dobj])
        input_neg2 = np.hstack([self.verb, self.pos_sbj, self.neg_dobj])
        input_neg3 = np.hstack([self.verb, self.neg_sbj, self.neg_dobj])
        return [input_pos, input_neg1, input_neg2, input_neg3], self.y


normal_func = lambda x: tf.exp(-x*x/2)

class Cruys2014(object):

    def __init__(self, num_hidden_layers=1, activation_func=tf.nn.tanh):
        inputs = [tf.keras.layers.Input(shape=(3,)) for i in range(4)]
        # "We set N, the size of our embedding matrices, to 50"
        emb = tf.keras.layers.Embedding(13000, 50)
        flatten = tf.keras.layers.Flatten()
        # "... and H, the number of units in the hidden layer, to 100."
        # "[...] f(·) represents the element-wise activation function tanh [...]"
        hiddens = [tf.keras.layers.Dense(100, activation=activation_func) 
                   for _ in range(num_hidden_layers)]
        # y = W_2*a_1 "is our final selectional preference score"
        score = tf.keras.layers.Dense(1, use_bias=False)

        def score_tuple_input(input_):
            intermediate_activation = flatten(emb(input_))
            for hidden in hiddens:
                intermediate_activation = hidden(intermediate_activation)
            return score(intermediate_activation)
        
        scores = [score_tuple_input(inp) for inp in inputs]    
        concat_scores = tf.keras.layers.Concatenate()(scores)  
        self.single_model = tf.keras.Model(inputs=inputs[0], outputs=scores[0])
        self.contrastive_model = tf.keras.Model(inputs=inputs, outputs=concat_scores)
        self.contrastive_model.compile(loss=Cruys2014.contrastive_loss, 
                                       metrics=[Cruys2014.contrastive_accuracy],
                                       optimizer='RMSprop')
        
    def fit(self, *args, **kwargs):
        return self.contrastive_model.fit(*args, **kwargs)
        
    def predict(self, *args, **kwargs):
        return self.single_model.predict(*args, **kwargs)
        
    def summary(self):
        print('=== Single model ===')
        self.single_model.summary()        
        print('=== Contrastive model ===')
        self.contrastive_model.summary()        

    @classmethod
    def contrastive_loss(cls, gold_dummy, predicted):
        pos_scores, neg_scores1, neg_scores2, neg_scores3 = \
                predicted[:,0], predicted[:,1], predicted[:,2], predicted[:,3]
        max_margin = lambda p, n: tf.reduce_mean(tf.maximum(0.0, 1 - p + n))
        return (max_margin(pos_scores, neg_scores1) +
                max_margin(pos_scores, neg_scores2) +
                max_margin(pos_scores, neg_scores3))

    @classmethod
    def contrastive_accuracy(cls, gold_dummy, predicted):
        pos_scores, neg_scores1, neg_scores2, neg_scores3 = \
                predicted[:,0], predicted[:,1], predicted[:,2], predicted[:,3]
        correct = pos_scores > neg_scores1
        correct = tf.logical_and(correct, pos_scores > neg_scores2)
        correct = tf.logical_and(correct, pos_scores > neg_scores3)
        return tf.reduce_mean(tf.cast(correct, tf.float32))