#!/usr/bin/python
import tensorflow as tf
import os

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

# os.environ["CUDA_VISIBLE_DEVICES"] = "12"

FLAGS = tf.app.flags.FLAGS #这个flag是个啥。好像是tensorflow中的函数。

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', './models/26.npy',
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', True,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase

    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            print(FLAGS.train_cnn)
            print(FLAGS.load_cnn)
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            # model.load_cnn_ckpt(sess,'./resnet_v1_50.ckpt')
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)


        elif FLAGS.phase == 'eval':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        else:
            # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)

if __name__ == '__main__': #如果当前执行的程序名称是main，那么就开始跑app接口。

    tf.app.run()
