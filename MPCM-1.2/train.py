from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, Matcher, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 768, "The output size of your model.") # originally 750. Should pass in to max_parag_len
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def load_data(path, mode):
    parag_id = []
    ques_id = []
    span = []
    parag = []
    ques = []
    ans = []
    with open(pjoin(FLAGS.data_dir, "{}.context".format(mode)), "r") as f:
        for line in f:
            parag.append(line.split())
    with open(pjoin(FLAGS.data_dir, "{}.question".format(mode)), "r") as g:
        for line in g:
            ques.append(line.split())
    with open(pjoin(FLAGS.data_dir, "{}.answer".format(mode)), "r") as h:
        for line in h:
            ans.append(line.split())

    with open(pjoin(FLAGS.data_dir, "{}.ids.context".format(mode)), "r") as i:
        for line in i:
            parag_id.append(map(int, line[:-1].split()))
    with open(pjoin(FLAGS.data_dir, "{}.ids.question".format(mode)), "r") as j:
        for line in j:
            ques_id.append(map(int, line[:-1].split()))
    with open(pjoin(FLAGS.data_dir, "{}.span".format(mode)), "r") as k:
        for line in k:
            span.append(map(int, line[:-1].split()))
    return parag_id, ques_id, span, parag, ques, ans

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = {"train": load_data(FLAGS.data_dir, mode="train"), \
               "val": load_data(FLAGS.data_dir, mode="val")}

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(state_size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    matcher = Matcher(perspective_dim=50, input_size=FLAGS.state_size) # add flag
    decoder = Decoder(output_size=FLAGS.output_size, state_size=FLAGS.state_size, n_perspective_dim=50*2) # add flag

    qa = QASystem(encoder, matcher, decoder, \
                  vocab=vocab, vocab_dim=FLAGS.embedding_size, rev_vocab=rev_vocab, embed_path=embed_path)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        tf.global_variables_initializer().run()
        graph_writer = tf.summary.FileWriter("qa-graph")
        graph_writer.add_graph(sess.graph)
        qa.train(sess, dataset, save_train_dir)
   
        qa.evaluate_answer(sess, dataset, 500, log=True)

if __name__ == "__main__":
    tf.app.run()
