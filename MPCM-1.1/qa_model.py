from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, random
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from collections import Counter
from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

class Config:
    max_q_len = 32
    max_p_len = 768
    max_grad_norm = 10.
    batch_size = 192
    max_epoch = 15
    dropout_keep_prob = 0.85
    embed_size = 300 # should adjust by retrieving embed_size directly

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
        ret.append((padded, masking, len(input_token)))
    return zip(*ret)

def iterate_across(padded_x, padded_y=None, batch_size=1):
    """
    Iterate across the padded data
    Args:
        padded_data: padded inputs and masks
        batch_size: number of training examples in a minibatch
    Yield:
        an iterator across the dataset
    """
    curr = 0
    data_len = len(padded_x[0])
    index_shuffler = list(range(data_len))
    random.shuffle(index_shuffler)
    while curr < data_len: # and curr < 1000: <-- for debugging purpose
        if padded_y is not None:
            yield [np.array([padded_x[i][j] \
                   for j in index_shuffler[curr:curr+batch_size]]) \
                   for i in range(len(padded_x))], \
                  [np.array([padded_y[i][j] \
                   for j in index_shuffler[curr:curr+batch_size]]) \
                   for i in range(len(padded_y))]
        else:
            yield [np.array([padded_x[i][j] \
                   for j in index_shuffler[curr:curr+batch_size]]) \
                   for i in range(len(padded_x))]
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
            cell_fw = tf.contrib.rnn.GRUCell(self.state_size)
            cell_bw = tf.contrib.rnn.GRUCell(self.state_size)
            (out_fw, out_bw), (fin_fw, fin_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=actual_len, dtype=tf.float32)
            out_fw = tf.nn.dropout(out_fw, keep_prob=Config.dropout_keep_prob)
            out_bw = tf.nn.dropout(out_bw, keep_prob=Config.dropout_keep_prob)
            fin_fw = tf.nn.dropout(fin_fw, keep_prob=Config.dropout_keep_prob)
            fin_bw = tf.nn.dropout(fin_bw, keep_prob=Config.dropout_keep_prob)
        return (out_fw, out_bw), (fin_fw, fin_bw) # (out_fwd, out_bwd)

class Matcher(object):
    def __init__(self, perspective_dim, input_size):
        self.perspective_dim = perspective_dim
        self.input_size = input_size
    def match(self, ps, p_finals, p_mask, qs, q_finals, q_mask):

        def batch_full_match(h1, h2_final, W):
            ## h1       : [None, time_step, input_size]
            ## h2_final : [None, input_size]
            ## W        : [input_size, perspective_d]
            ## m        : [None, time_step, perspective_d]
            #W = tf.expand_dims(tf.expand_dims(W, 0), 1)
            #h1 = tf.expand_dims(h1, 3)
            #h2_final = tf.expand_dims(tf.expand_dims(h2_final, 1), 3)
            #first = W * h1  # None x T x input_size x perspective_d
            #second = W * h2_final # None x 1 x input_size x perspective_d --> none per T 1
            #inner = tf.transpose(tf.matmul(tf.transpose(first,perm=[0,3,1,2]), tf.transpose(second,perm=[0,3,2,1])), perm=[0,2,3,1])
            #norms = tf.norm(first, axis=2, keep_dims=True) * tf.norm(second, axis=2, keep_dims=True) # none t 1 oer
            #return tf.squeeze((inner + 1e-7) / (norms + 1e-7), [2])
            target_shape = [-1, tf.shape(h1)[1], self.perspective_dim]
            inner = tf.reshape(tf.matmul(tf.reshape(h1 * tf.expand_dims(h2_final, 1), [-1,self.input_size]), W * W), target_shape)
            h1_reshape = tf.reshape(h1, [-1,self.input_size])
            norms = tf.reshape(tf.sqrt(tf.matmul(h1_reshape*h1_reshape + .0001, W*W)), target_shape) * tf.expand_dims(tf.sqrt(tf.matmul(h2_final*h2_final + .0001, W*W)), 1)
            return inner / norms
        with tf.variable_scope("matcher"):
            initializer = tf.contrib.layers.xavier_initializer()
            W1 = tf.Variable(initializer([self.input_size, self.perspective_dim]))
            W2 = tf.Variable(initializer([self.input_size, self.perspective_dim]))
            p_fw, p_bw = ps # None x time_step x input_size
            q_final_fw, q_final_bw = q_finals # None x input_size
            match_fw = batch_full_match(p_fw, q_final_fw, W1) # None x time_step x perspective_d
            match_bw = batch_full_match(p_bw, q_final_bw, W2) # None x time_step x perspective_d
            return tf.nn.dropout(tf.concat([match_fw, match_bw], axis=2), keep_prob=Config.dropout_keep_prob)
            # None x time_step x 2*perspective_d


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
        with tf.variable_scope("decoder", initializer=tf.contrib.layers.xavier_initializer()):
            initializer = tf.contrib.layers.xavier_initializer()
            cell_fw = tf.contrib.rnn.GRUCell(self.state_size)
            cell_bw = tf.contrib.rnn.GRUCell(self.state_size)
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, knowledge_rep, sequence_length=actual_len, dtype=tf.float32) # [None x time_step x size]
            outputs = tf.nn.dropout(tf.concat([out_fw, out_bw], axis=2), keep_prob=Config.dropout_keep_prob)
            W = tf.Variable(initializer([self.state_size * 2, 2])) # assert time_step == output_size
            b = tf.Variable(tf.zeros([1, self.output_size, 2]))
            score = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, self.state_size*2]), W), [-1, self.output_size, 2]) + b
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
        optimizer = get_optimizer("adam")() # figure out a way to use adam
        # Gradient clipping
        grads_and_vars = optimizer.compute_gradients(self.loss)
        grads, tvars = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, Config.max_grad_norm)
        self.grads = grads
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def get_feed_dict(self, batch_x, batch_y=None):
        p, p_mask, p_actual_len, q, q_mask, q_actual_len = batch_x
        feed_dict = {
            self.p_placeholder: p,
            self.p_mask_placeholder: p_mask,
            self.p_actual_len_placeholder: p_actual_len,
            self.q_placeholder: q,
            self.q_mask_placeholder: q_mask,
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
        # p = batch x p_time_step x dim
        # q = batch x q_time_step x dim
        # r = batch x p 
        qt = tf.transpose(self.q, perm=[0,2,1])
        inner = tf.matmul(self.p, qt)
        norms = tf.matmul(tf.sqrt(tf.reduce_sum(self.p*self.p, axis=2, keep_dims=True)+0.0001), tf.sqrt(tf.reduce_sum(qt, axis=1, keep_dims=True)+.0001))
        self.pf = p_filt = self.p * tf.reduce_max(inner / norms, axis=2, keep_dims=True)
        with tf.variable_scope("encoder") as scope:
            p_outs, p_finals = self.encoder.encode(p_filt, self.p_mask_placeholder, self.p_actual_len_placeholder)
            scope.reuse_variables()
            q_outs, q_finals = self.encoder.encode(self.q, self.q_mask_placeholder, self.q_actual_len_placeholder)
        self.m = m_out = self.matcher.match(p_outs, p_finals, self.p_mask_placeholder, q_outs, q_finals, self.q_mask_placeholder)
        scores = self.decoder.decode(m_out, self.p_actual_len_placeholder)
        self.begin_score = scores[:,:,0]
        self.end_score = scores[:,:,1]

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with tf.variable_scope("loss"):
            loss_begin = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.begin_placeholder, logits=self.begin_score)
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.end_placeholder, logits=self.end_score)
            self.loss = tf.reduce_mean(loss_begin + loss_end)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"), tf.device("/cpu:0"):
            embedding_tensor = tf.Variable(np.load(self.embed_path)["glove"].astype(np.float32), trainable=False) # should be true
            self.p = tf.nn.embedding_lookup(embedding_tensor, self.p_placeholder)
            self.q = tf.nn.embedding_lookup(embedding_tensor, self.q_placeholder)

    def optimize(self, sess, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.get_feed_dict(train_x, train_y)
        output_feed = [self.loss, self.grads, self.p, self.q, self.train_op]
        loss, gr, p, q,  _ = sess.run(output_feed, input_feed)
        # print("LOSS = ",loss)
        # print("p norm", np.linalg.norm(p))
        # print("q norm", np.linalg.norm(q))
        # for i in range(1,len(gr)):
        #    print(gr[i].shape, np.sum(np.isnan(gr[i])), np.linalg.norm(gr[i]))
        # print(gr[0])
        return loss

    def test(self, sess, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = self.get_feed_dict(valid_x, valid_y)
        output_feed = [self.loss]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0]

    def decode(self, sess, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self.get_feed_dict(test_x)
        output_feed = [self.begin_score, self.end_score, self.begin_score]
        begin_out, end_out, x = sess.run(output_feed, input_feed)
        return begin_out, end_out

    def answer(self, sess, test_x):
        yp, yp2 = self.decode(sess, test_x)
        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)
        return (a_s, a_e)

    def answer_all(self, sess, test_x):
        """
        returned as list
        """
        all_a_s = []
        all_a_e = []
        for batch_x in iterate_across(padded_x=test_x, batch_size=Config.batch_size):
            (a_s, a_e) = self.answer(sess, batch_x)
            all_a_s += list(a_s)
            all_a_e += list(a_e)
        return all_a_s, all_a_e

    def validate(self, sess, valid_x, valid_y):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.
        This method calls self.test() which explicitly calculates validation cost.
        How you implement this function is dependent on how you design
        your data iteration function
        :return:
        """
        valid_loss = 0
        num_iter = 0
        for batch_x, batch_y in iterate_across(valid_x, valid_y, Config.batch_size):
            valid_loss += self.test(sess, batch_x, batch_y)
            num_iter += 1
        return valid_loss / num_iter

    def evaluate_answer(self, sess, dataset, sample=100, log=False, mode="val"):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels
        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.
        :param sess: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = 0.
        em = 0.
        len_data = len(dataset[mode][-1])
        indexer = random.sample(xrange(len_data), sample)
        # 1. Pad data
        p, q, span = [[component[i] for i in indexer] for component in dataset[mode]]
        p, mask_p, actual_p = pad_input(p, Config.max_p_len)
        q, mask_q, actual_q = pad_input(q, Config.max_q_len)
        begin, end = zip(*span)
        # get answer
        a_s, a_e = self.answer_all(sess, [p, mask_p, actual_p, q, mask_q, actual_q])
        for i in range(sample):
            # EM
            em += 1. if a_s[i] == begin[i] and a_e[i] == end[i] else 0.
            # F1
            pred_ans = p[i][a_s[i] : a_e[i]+1]
            actual_ans = p[i][begin[i] : end[i]+1]
            common = Counter(pred_ans) & Counter(actual_ans)
            num_same = sum(common.values())
            if num_same > 0:
                f1 += 2.0 * (num_same) / (len(pred_ans) + len(actual_ans))
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1/sample , em/sample, sample))

        return f1 / sample, em / sample

    def train_epoch(self, sess, data_x, data_y):
        train_loss = 0
        num_iter = 0
        prev = 0
        tic = time.time()
        for batch_x, batch_y in iterate_across(data_x, data_y, Config.batch_size):
            this_train_loss = self.optimize(sess, batch_x, batch_y)
            train_loss += this_train_loss
            num_iter += 1
            if num_iter >= prev * 2:
                print("Iteration {} / {}, Average time per iteration = {} s, ETA = {}".format( \
                num_iter, (len(data_y[0]) - 1) // Config.batch_size + 1,(time.time()-tic) / num_iter,\
                (time.time()-tic) * ((len(data_y[0]) - 1) // Config.batch_size + 1 - num_iter) / num_iter))
                prev = num_iter
        return train_loss / num_iter


    def train(self, sess, dataset, train_dir):
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
        :param sess: it should be passed in from train.py
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
        p_val, q_val, span_val = dataset["val"]
        p_val, mask_p_val, actual_p_val = pad_input(p_val, Config.max_p_len)
        q_val, mask_q_val, actual_q_val = pad_input(q_val, Config.max_q_len)
        begin_val, end_val = zip(*span_val)
        # 3. iterate, run_epoch
        print("Number of params: {}".format(num_params))
        for ep in range(Config.max_epoch):
            tic_epoch = time.time()
            print("Epoch {}".format(ep))
            train_loss = self.train_epoch(sess, [p, mask_p, actual_p, q, mask_q, actual_q], [begin, end])
            f1_train, em_train = self.evaluate_answer(sess, dataset, sample=500, log=False, mode='train')
            print("Train loss: {} Train F1: {} Train EM: {}".format(train_loss, f1_train, em_train))
            val_loss = self.validate(sess, [p_val, mask_p_val, actual_p_val, q_val, mask_q_val, actual_q_val], [begin_val, end_val])
            f1_val, em_val = self.evaluate_answer(sess, dataset, sample=500, log=False, mode='val')
            print("Val loss: {} Val F1: {} Val EM: {}".format(val_loss, f1_val, em_val))
            print("Time taken: {} s".format(time.time() - tic_epoch))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
