
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'vgg16'               # 'vgg16' or 'resnet50'
        # self.cnn = 'resnet50'
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 1      # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 1024
        self.bn = True

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 1
        self.batch_size = 10
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        # self.save_period = 1000
        self.save_period = 10000
        self.save_dir = './models/'
        self.summary_dir = './summary/'

        """Sydney dataset"""
        # about the vocabulary
        self.vocabulary_file = './vocabulary.csv'
        self.vocabulary_size = 210
        # about the training
        self.train_image_dir = './train/images/'
        self.train_caption_file = './train/captions_train2014.json'#这个地方需要改变路径
        self.temp_annotation_file = './train/anns.csv'
        self.temp_data_file = './train/data.npy'

        # about the evaluation
        self.eval_image_dir = './train/images/'
        self.eval_caption_file = './data_val_coco.json'
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = True



        # """UCM dataset"""
        # # about the vocabulary
        # self.vocabulary_file = './vocabulary.csv'
        # self.vocabulary_size = 280
        #
        # # about the training
        # self.train_image_dir = './UCM_Captions/imgs/'
        # self.train_caption_file = './UCM_Captions/data_train_UCM.json'#这个地方需要改变路径
        # self.temp_annotation_file = './UCM_Captions/anns.csv'
        # self.temp_data_file = './UCM_Captions/data.npy'
        #
        # # about the evaluation
        # self.eval_image_dir = './UCM_Captions/imgs/'
        # self.eval_caption_file = './UCM_Captions/data_val_UCM.json'
        # self.eval_result_dir = './val/results/'
        # self.eval_result_file = './val/results.json'
        # self.save_eval_result_as_image = False
        #
        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results/results.csv'

        self.alpha_temp=[]
