from data import Indexer
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adagrad import AdagradOptimizer
from hyperspace.utils import read_train_dataset, read_dev_dataset

tf.set_random_seed(125)
batch_size = 100000

class VanDeCruys2014Model(object):
    '''Implement Section 3.1 and 3.2 in van de Cruys (2014).
    
    Notice: because we use two different vocabularies for contexts (words or
    combination of words and dependency labels) and target words (just words),
    we need to divide W1 into two parts: W1_cxt and W1_tar.
    '''
    
    def __init__(self, cxt_vocab_size, tar_vocab_size, emb_dims, hidden_size):
        
        self.cxt_vocab_size = cxt_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.hidden_size = hidden_size
        
        self.cxt_embeddings = tf.Variable(
                tf.random_uniform([cxt_vocab_size, emb_dims], -1.0, 1.0))
        self.tar_embeddings = tf.Variable(
                tf.random_uniform([tar_vocab_size, emb_dims], -1.0, 1.0))
        self.W1_cxt = tf.Variable(
                tf.random_uniform([emb_dims, hidden_size], -0.01, 0.01))
        self.W1_tar = tf.Variable(
                tf.random_uniform([emb_dims, hidden_size], -0.01, 0.01))
        self.b1 = tf.Variable(tf.random_uniform([1, hidden_size], -0.01, 0.01))
        self.W2 = tf.Variable(tf.random_uniform([hidden_size, 1], -0.01, 0.01))
        
        self.x = tf.placeholder(tf.int32, shape=[None])
        self.y_pos = tf.placeholder(tf.int32, shape=[None])
        self.y_neg = tf.placeholder(tf.int32, shape=[None])
        
        score_pos = self._forward(self.x, self.y_pos)
        score_neg = self._forward(self.x, self.y_neg)
        
        # "We require the score for the correct tuple to be larger than 
        # the score for the corrupt tuple by a margin of one."
        self.loss = tf.reduce_sum(tf.maximum([0.0], 1-score_pos+score_neg))
        opt = AdagradOptimizer(learning_rate=0.1)
        self.params = [self.cxt_embeddings, self.tar_embeddings, 
                       self.W1_cxt, self.W1_tar, self.b1, self.W2]
        self.opt_op = opt.minimize(self.loss, var_list=self.params)
        self.acc = tf.reduce_mean(tf.cast(score_pos > score_neg, dtype=tf.float32))

    def _forward(self, x, y):
        x_emb = tf.nn.embedding_lookup(self.cxt_embeddings, x)
        y_emb = tf.nn.embedding_lookup(self.tar_embeddings, y)
        z_cxt = tf.matmul(x_emb, self.W1_cxt)
        z_tar = tf.matmul(y_emb, self.W1_cxt)
        a1 = tf.nn.tanh(z_cxt + z_tar + self.b1)
        return tf.matmul(a1, self.W2)

    def fit(self, sess, x_val, y_pos_val, num_batches=100):
        sess.run(tf.global_variables_initializer())
        for batch_no in range(num_batches):
            batch_indices = np.random.choice(x_val, size=batch_size, replace=False)
            batch_x_pos = x_val[batch_indices]
            batch_y_pos = y_pos_val[batch_indices]
            batch_y_neg = np.random.choice(self.tar_vocab_size, size=batch_size)
            sess.run(self.opt_op, feed_dict={self.x: batch_x_pos, 
                                             self.y_pos: batch_y_pos,
                                             self.y_neg: batch_y_neg})
            if (batch_no+1) % 10 == 0:
                print("Batch #%d ..." %(batch_no+1))

    def eval(self, sess, x_val, y_pos_val, y_neg_val):
        acc = sess.run(self.acc, feed_dict={self.x: x_val, 
                                             self.y_pos: y_pos_val,
                                             self.y_neg: y_neg_val})
        print("Accuracy: %.2f%%" %(acc*100))

def run():
    ''' Put everything here to avoid contaminating global scope. '''
    cxt_indexer, tar_indexer = Indexer(), Indexer()
    train_x, train_y_pos = read_train_dataset('output/dep_context_labeled.train.txt', 
                                              cxt_indexer, tar_indexer)
    cxt_indexer.seal(True)
    tar_indexer.seal(True)
    dev_x, dev_y_pos, dev_y_neg = read_dev_dataset('output/dep_context_labeled.dev-fixed.txt', 
                                                   cxt_indexer, tar_indexer)

    # "We set N, the size of our embedding matrices, to 50, 
    # and H, the number of units in the hidden layer, to 100."
    model = VanDeCruys2014Model(len(cxt_indexer), len(tar_indexer), 50, 100)
    with tf.Session() as sess:
        model.fit(sess, train_x, train_y_pos, num_batches=100)
        model.eval(sess, dev_x, dev_y_pos, dev_y_neg)
    
        
if __name__ == '__main__':
    run()