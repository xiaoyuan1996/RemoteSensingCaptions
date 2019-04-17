import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow
#import cPickle as pickle
import pickle
import copy
import json
from tqdm import tqdm

from utils.nn import NN
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN
from config import Config

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.global_epoch = 0
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data):
        """ Train the model using the COCO train2014 data. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        Loss_table = []
        Loss_line = []
        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                image_files, sentences, masks = batch
                images = self.image_loader.load_images(image_files)
                feed_dict = {self.images: images,
                             self.sentences: sentences,
                             self.masks: masks}
                Loss, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step],
                                                    feed_dict=feed_dict)
                Loss_line.append(Loss)
                # if (global_step + 1) % config.save_period == 0:
                #     self.save()
                # train_writer.add_summary(summary, global_step)
                if (global_step + 1) % 10 == 0:
                    train_writer.add_summary(summary, global_step)
            Loss_table.append(Loss_line)
            Loss_line = []
            self.save()
            train_data.reset()

        file = open('ResultProcess/Loss/Loss.pickle', 'wb')
        pickle.dump(Loss_table, file)
        file.close()

        self.save()
        train_writer.close()
        print("Training complete.")

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

########################################################################################
        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            batch = eval_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                results.append({'image_id': int(eval_data.image_ids[idx]),
                                'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(
                                            # config.eval_result_dir,
                                             image_name+'_result.jpg'))
        fp = open(config.eval_result_file, 'w')
        json.dump(results, fp)
        #json_dict = json.dumps(results)
        #fp.write(json_dict)
        fp.close()
#######################################################################################################
        print('json done')
        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                # Save the result in an image file
                image_file = batch[l]
                image_name = image_file.split(os.sep)[-1]
                image_name = os.path.splitext(image_name)[0]
                img = plt.imread(image_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(caption)
                plt.savefig(os.path.join(
                                         config.test_result_dir,
                                         image_name[-1]+'_result.jpg'))

        # Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")

    def beam_search(self, sess, image_files, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        images = self.image_loader.load_images(image_files)
        contexts, initial_memory, initial_output   = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output ],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []

        alpha_all = []
        last_output_all = []

        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        # Run beam search
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores  = sess.run(
                    [self.memory, self.output, self.probs],
                    feed_dict = {self.contexts: contexts,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output})


                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:config.beam_size+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

            results = []

            alpha_status = sess.run(self.alpha,
                                    feed_dict={self.contexts: contexts,
                                               self.last_memory: last_memory,
                                               self.last_output: last_output})

            alpha_all.append(alpha_status)
            last_output_all.append(last_output)


        file = open('ResultProcess/alpha.pickle', 'wb')
        pickle.dump(alpha_all, file)
        file.close()
        # file = open('ResultProcess/last_output_all.pickle', 'wb')
        # pickle.dump(last_output_all, file)
        # file.close()



        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))


        return results

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_epoch))
        self.global_epoch += 1

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path,encoding='latin1').item()
        del data_dict['fc6']
        del data_dict['fc7']
        del data_dict['fc8']

        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                # for param_name, data in data_dict[op_name].iteritems():
                # for param_name, data in data_dict[op_name].items():
                # for param_name, data in data_dict[op_name]:
                #     try:
                #         var = tf.get_variable(param_name)
                #         session.run(var.assign(data))
                #         count += 1
                #     except ValueError:
                #         pass
                data = data_dict[op_name]
                try:
                    var_k = tf.get_variable('kernel')
                    session.run(var_k.assign(data[0]))
                    var_b = tf.get_variable('bias')
                    session.run(var_b.assign(data[1]))
                    count += 1
                except ValueError as e:
                    assert e
                    pass
        print("%d tensors loaded." %count)

    def load_cnn_ckpt(self, session, data_path):
        print("Loading the CNN from %s..." % data_path)
        reader = pywrap_tensorflow.NewCheckpointReader(data_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        del var_to_shape_map['global_step']
        del var_to_shape_map['resnet_v1_50/mean_rgb']
        del var_to_shape_map['resnet_v1_50/logits/weights']
        del var_to_shape_map['resnet_v1_50/logits/biases']
        count = 0
        for key in tqdm(var_to_shape_map):
            ckpt_tensor = reader.get_tensor(key)
            tensor_name = self.get_tensor_name(key.split('/')).split('/')
            with tf.variable_scope(tensor_name[0], reuse = True):
                var_ = tf.get_variable(tensor_name[1])
                session.run(var_.assign(ckpt_tensor))
                count += 1
        with tf.variable_scope('conv1', reuse = True):
            var_ = tf.get_variable('bias')
            session.run(var_.assign(np.zeros([64])))
            count += 1
        print("%d tensors loaded." % count)

    def get_tensor_name(self,keys):
        name = ''
        flag = False
        n = 0
        for item in ['block1','block2','block3','block4']:
            if item in keys:
                flag = True
            if flag:
                if 'BatchNorm' in keys:
                    name += 'bn' + str(n + 2)
                else:
                    name += 'res' +  str(n + 2)
                break
            elif n == 3:
                if 'BatchNorm' in keys:
                    name += 'bn_conv1/'
                else:
                    name += 'conv1/'
            n += 1

        flag = False
        n = 0
        for item in ['unit_1','unit_2','unit_3','unit_4','unit_5','unit_6']:
            if item in keys:
                flag = True
            if flag:
                name += chr(97 + n) + '_'
                break
            n += 1

        flag = False
        n = 0
        for item in ['shortcut','conv1','conv2','conv3']:
            if item in keys:
                flag = True
            if flag and (name[-2:-7:-1] != '1vnoc'):
                if n == 0:
                    name += 'branch1'
                else:
                    name += 'branch2' + chr(96 + n)
                name += '/'
                break
            n += 1

        flag = False
        n = 0
        for item in ['weights','gamma','beta','moving_mean','moving_variance']:
            if item in keys:
                flag = True
            if flag:
                if n == 0:
                    name += 'kernel'
                else:
                    name += item
                break
            n += 1

        return name