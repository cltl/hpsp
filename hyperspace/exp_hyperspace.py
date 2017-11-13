from data import Indexer
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adagrad import AdagradOptimizer
from hyperspace.utils import read_train_dataset, read_dev_dataset

batch_size = 100000

class HyperspaceModel(object):
    
    def __init__(self, cxt_vocab_size, tar_vocab_size, emb_dims):
        self.cxt_vocab_size = cxt_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.emb_dims = emb_dims
        
        self.hyperspace_coeffs = tf.Variable(
                tf.random_uniform([cxt_vocab_size, emb_dims+1], -1.0, 1.0))
        self.word_embeddings = tf.Variable(
                tf.random_uniform([tar_vocab_size, emb_dims], -1.0, 1.0))
        self.x = tf.placeholder(tf.int32, shape=[None])
        self.y_pos = tf.placeholder(tf.int32, shape=[None])
        self.y_neg = tf.placeholder(tf.int32, shape=[None])
        
        x_coeffs = tf.nn.embedding_lookup(self.hyperspace_coeffs, self.x)
        y_neg_emb = tf.nn.embedding_lookup(self.word_embeddings, self.y_neg)
        y_pos_emb = tf.nn.embedding_lookup(self.word_embeddings, self.y_pos)
        score_pos = tf.abs(tf.reduce_sum(x_coeffs[:,:-1]*y_pos_emb, axis=1) + x_coeffs[:,-1])
        score_neg = tf.abs(tf.reduce_sum(x_coeffs[:,:-1]*y_neg_emb, axis=1) + x_coeffs[:,-1])
        
        self.loss = score_pos - tf.log(1 - tf.exp(-score_neg))
        opt = AdagradOptimizer(learning_rate=0.1)
        self.opt_op = opt.minimize(self.loss, var_list=[self.hyperspace_coeffs,
                                                        self.word_embeddings])
        self.acc = tf.reduce_mean(tf.cast(score_pos < score_neg, dtype=tf.float32))

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
            if (batch_no+1) % 100 == 0:
                print("Batch #%d ..." %(batch_no+1))

    def eval(self, sess, x_val, y_pos_val, y_neg_val):
        acc = sess.run(self.acc, feed_dict={self.x: x_val, 
                                             self.y_pos: y_pos_val,
                                             self.y_neg: y_neg_val})
        print("Accuracy: %.2f%%" %(acc*100))

def run(seed):
    ''' Put everything here to avoid contaminating global scope. '''
    print('Training and evaluating the hyperspace model with random seed %d' %seed)
    tf.set_random_seed(seed)

    cxt_indexer, tar_indexer = Indexer(), Indexer()
    train_x, train_y_pos = read_train_dataset('output/dep_context_labeled.train.txt', 
                                              cxt_indexer, tar_indexer)
    cxt_indexer.seal(True)
    tar_indexer.seal(True)
    dev_x, dev_y_pos, dev_y_neg = read_dev_dataset('output/dep_context_labeled.dev-fixed.txt', 
                                                   cxt_indexer, tar_indexer)

    model = HyperspaceModel(len(cxt_indexer), len(tar_indexer), 100)
    with tf.Session() as sess:
        model.fit(sess, train_x, train_y_pos, num_batches=1000)
        model.eval(sess, dev_x, dev_y_pos, dev_y_neg)
    
        
if __name__ == '__main__':
    for seed in [19, 27, 572, 957, 726, 62, 50, 275, 827, 295]:
        run(seed)