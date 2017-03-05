.p_maskfrom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

class Config:
   max_q_len = 24
   max_p_len = 264
   batch_size = 64
   max_epoch = 20
   dropout_keepprob = 0.5
   embed_size = 100 # should adjust by retrieving embed_size directly

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def pad_input(unpadded_data, max_length):
    """
    Pad/cap input to max_length
    Args:
        unpadded_data: a list of input tokens of different lengths. Can be a list
              of questions or paragraphs.
              Question and paragraphs are represented by a list of numeric ids.
        max_length: maximum length of the input token. Must be specified in config
              Generally we should estimate that number before training by looking
              at the data. In the config, there should be both max_question_len
              and max_paragraph_len. Use those to pass in as this max_length.
    Returns:
        zip(*ret) : ret is a list of tuples [(padded, mask), (padded2, mask2), ...]
             Each tuples contain masked input and the mask to be applied.
    """
    ret = []
    for input_token in unpadded_data:
        diff = max_length - len(input_token)
        if diff > 0:
            padded = input_token + diff * [0]
            masking = [True] * len(input_token) + [False] * diff
        else:
            padded = input_token[:max_length]
            masking = [True] * max_length
            actual = len(input_token)
        ret.append((padded, masking, actual))
    return zip(*ret)

def iterator_across(padded_x, padded_y, batch_size):
    """
    Iterate across the padded data
    Args:
        padded_data: padded inputs and masks
        batch_size: number of training examples in a minibatch
    Yield:
        an iterator across the dataset
    """
    curr = 0
    data_len = len(padded_y[0])
    index_shuffler = list(range(data_len))
    random.shuffle(index_shuffler)
    while curr < data_len:
        yield [np.array([padded_x[i][j] \
               for j in index_shuffler[curr:curr+batch_size]]) \
               for i in range(len(padded_x))], \
              [np.array([padded_y[i][j] \
               for j in index_shuffler[curr:curr+batch_size]]) \
               for i in range(len(padded_y))]
        curr += batch_size

class Encoder(object):
    def __init__(self, state_size, vocab_dim):
        self.state_size = state_size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, actual_len):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with tf.variable_scope("encoder"):
            cell_fw = tf.contrib.rnn.GRUCell(state_size, input_size=vocab_dim)
            cell_bw = tf.contrib.rnn.GRUCell(state_size, input_size=vocab_dim)
            outputs, finals = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=actual_len)
        return outputs, finals # (out_fwd, out_bwd)

class Matcher(object):
    def __init__(self, perspective_dim, input_size):
        self.perspective_dim = perspective_dim
        self.input_size = input_size
    def match(self, ps, p_finals, qs, q_finals):
        def batch_full_match(batch_h1, h2, W):
            return
        with tf.variable_scope("matcher", initializer=tf.contrib.layers.xavier_initializer()):
            W1 = tf.Variable(initializer([input_size, perspective_dim]))
            W2 = tf.Variable(initializer([input_size, perspective_dim]))
            p_fw, p_bw = ps # None x time_step x input_size
            q_final_fw, q_final_bw = q_finals # None x input_size
            match_fw = batch_full_match(p_fw, q_final_fw, W1) # None x time_step x perspective_d
            match_bw = batch_full_match(p_bw, q_final_bw, W2) # None x time_step x perspective_d
            return tf.concat(match_fw, match_bw, axis=2) # None x time_step x 2*perspective_d


class Decoder(object):
    def __init__(self, output_size, state_size, n_perspective_dim):

        self.output_size = output_size
        self.state_size = state_size
        self.n_perspective_dim = n_perspective_dim

    def decode(self, knowledge_rep, actual_len):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder [None x time_step x n_perspective_dim]
        :return:
        """
        with tf.variable_scope("decoder", initializer=tf.contrib.layers.xavier_initializer())
            cell_fw = tf.contrib.rnn.GRUCell(state_size, input_size=n_perspective_dim)
            cell_bw = tf.contrib.rnn.GRUCell(state_size, input_size=n_perspective_dim)
            outputs, _ = tf.nn_bidirectional_dynamic_rnn(cell_fw, cell_bw, knowledge_rep, sequence_length=actual_len) # [None x time_step x size]
            W = tf.Variable(initializer([state_size, 2]))
            b = tf.Variable(tf.zeros([1, output_size, 2]))
            score = tf.squeeze(tf.matmul(outputs, W), [2]) + b
        return score # [None x output_size] = [None x time_step x 2]

class QASystem(object):
    def __init__(self, encoder, matcher, decoder, **kwargs):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.embed_path = kwargs["embed_path"]
        self.embed_size = kwargs["vocab_dim"]
        self.encoder = encoder
        self.matcher = matcher
        self.decoder = decoder
        # ==== set up placeholder tokens ========
        self.p_placeholder = tf.placeholder(tf.int32, [None, Config.max_p_len])
        self.p_mask_placeholder = tf.placeholder(tf.bool, [None, Config.max_p_len])
        self.p_actual_len_placeholder = tf.placeholder(tf.int32, [None])
        self.q_placeholder = tf.placeholder(tf.int32, [None, Config.max_q_len])
        self.q_mask_placeholder = tf.placeholder(tf.bool, [None, Config.max_q_len])
        self.q_actual_len_placeholder = tf.placeholder(tf.int32, [None])
        self.begin_placeholder = tf.placeholder(tf.int32, [None])
        self.end_placeholder = tf.placeholder(tf.int32, [None])

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.contrib.layers.xavier_initializer()):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass

    def get_feed_dict(self, batch_x, batch_y=None):
        p, p_mask, p_actual_len, q, q_mask, q_actual_len = batch_x
        feed_dict = {
            self.p_placeholder: p,
            self.mask_p_placeholder: p_mask,
            self.p_actual_len_placeholder: p_actual_len,
            self.q_placeholder: q,
            self.q_mask_placeholder: q_mask
            self.q_actual_len_placeholder: q_actual_len,
        }
        if batch_y is not None:
            feed_dict[self.begin_placeholder] = batch_y[0]
            feed_dict[self.end_placeholder] = batch_y[1]
        return feed_dict

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        p, q = self.get_embeddings()
        p_mask = self.p_mask_placeholder
        q_mask = self.q_mask_placeholder
        p_outs, p_finals = self.encoder.encode(p, self.p_mask_placeholder, self.p_actual_len_placeholder)
        q_outs, q_finals = self.encoder.encode(q, self.q_mask_placeholder, self.q_actual_len_placeholder)
        m_out = self.matcher.match(p_outs, p_finals, q_outs, q_finals)
        self.scores = self.decoder.decode(m_out, self.p_actual_len_placeholder)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with tf.variable_scope("loss"):
            

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            embedding_tensor = tf.Variable(np.load(self.embed_path)["glove"])
            p_embeddings = tf.nn.embedding_lookup(embedding_tensor, self.p_placeholder)
            p_embeddings = tf.reshape(p_embeddings, [-1, Config.max_p_len, self.embed_size])
            q_embeddings = tf.nn.embedding_lookup(embedding_tensor, self.q_placeholder)
            q_embeddings = tf.reshape(q_embeddings, [-1, Config.max_q_len, self.embed_size])
        return p_embeddings, q_embeddings

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.get_feed_dict(train_x, train_y)
        opt = get_optimizer("adam")()
        # Gradient clipping here !!!
        train_op = opt.minimize(self.loss)
        output_feed = [train_op]
        outputs = session.run(output_feed, input_feed)

        return outputs

    #def test(self, session, valid_x, valid_y):
    #    """
    #    in here you should compute a cost for your validation set
    #    and tune your hyperparameters according to the validation set performance
    #    :return:
    #    """
    #    input_feed = {}

    #    # fill in this feed_dictionary like:
    #    # input_feed['valid_x'] = valid_x

    #    output_feed = []

    #    outputs = session.run(output_feed, input_feed)

    #    return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    #def validate(self, sess, valid_dataset):
    #    """
    #    Iterate through the validation dataset and determine what
    #    the validation cost is.

    #    This method calls self.test() which explicitly calculates validation cost.

    #    How you implement this function is dependent on how you design
    #    your data iteration function

    #    :return:
    #    """
    #    valid_cost = 0

    #    for valid_x, valid_y in valid_dataset:
    #      valid_cost = self.test(sess, valid_x, valid_y)


    #    return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def run_epoch(self, session, data_x, data_y, **kwargs ):
        for batch_x, batch_y in iterate_across(data_x, data_y, Config.batch_size):
            ?, ?, ?, ? = self.optimize(..., ..) # sess.run([cost, pred, eval_op], feed)


    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))

        # 1. Pad train data
        p, q, span = dataset["train"]
        p, mask_p, actual_p = pad_input(p, Config.max_p_len)
        q, mask_q, actual_q = pad_input(q, Config.max_q_len)
        begin, end = zip(*span)
        # 2. Pad val data

        # 3. iterate, run_epoch
        #????? = self.run_epoch(session, [p, mask_p, actual_p q, mask_q,, actual_q], [begin, end], ...)
        # use self.optimize

        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
