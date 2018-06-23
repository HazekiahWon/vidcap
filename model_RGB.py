#-*- coding: utf-8 -*-
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import pad_sequences
from bleu_eval import BLEU

# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#=====================================================================================
# Global Parameters
#=====================================================================================

video_train_feat_path = './features'
video_test_feat_path = './features'

video_train_data_path = './data/video_corpus.csv'
video_test_data_path = './data/video_corpus.csv'

time_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())
model_path = os.path.join('models', time_id)
logdir = os.path.join('tensorboard', time_id)

prompts = []
train_test_ratio = 0.9
scheduled_sampling = False
#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096
dim_hidden= 512

n_video_lstm_step = 80
n_caption_lstm_step = 30
n_frame_step = 80

n_epochs = 1000
batch_size = 32
# factor = 31
summ_freq = 25
save_freq = 50
learning_rate = 0.0001
alpha_baseline = 0.3*n_caption_lstm_step / float(n_video_lstm_step)
alph_loss_weight = 0.5

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, scheduled_sampling, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        # self.batch_size = batch_size

        self.n_frames=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"): # word embeddings
            self.word_embeddings = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)

        self.vis_encode_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.vis_encode_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')
        # print(self.encode_image_W.dtype)

        self.dense_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.dense_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.dense_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        ##========= switch ===========##
        self.use_scheduled_sampling = scheduled_sampling
        prompts.append('!!! scheduled_sampling {}'.format(self.use_scheduled_sampling))
        self.luong = False
        prompts.append('!!! luong {}'.format(self.luong))

        if self.luong:
            self.att_W = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='att_W')
            self.att_b = tf.Variable(tf.zeros([dim_hidden]), name='att_b')
        else :
            self.bah_w1 = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='bah_w1')
            self.bah_w2 = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='bah_w2')
            self.bah_v = tf.Variable(initial_value=tf.constant(np.math.sqrt(1./dim_hidden), dtype=tf.float32, shape=[dim_hidden]), name='bah_v')
#tf.Variable(tf.random_uniform([dim_hidden], -0.1, 0.1), name='bah_v')


    def _params_usage(self):
        total = 0
        for v in tf.trainable_variables():
            shape = v.get_shape()
            cnt = 1
            for dim in shape:
                cnt *= dim.value
            prompts.append('{} with shape {} has {}'.format(v.name, shape, cnt))
            total += cnt
        prompts.append('totaling {}'.format(total))

        # input()


    def build_model(self, train=True):

        # inputs
        vgg_vis_features = tf.placeholder(tf.float32, [None, self.n_frames, self.dim_image], name='video')
        video_mask = tf.placeholder(tf.float32, [None, self.n_frames], name='vid_mask')

        caption = None
        caption_mask = None
        if train:
            caption = tf.placeholder(tf.int32, [None, self.n_caption_lstm_step+1], name='caption')
            caption_mask = tf.placeholder(tf.float32, [None, self.n_caption_lstm_step+1], name='caption_mask')

        if self.use_scheduled_sampling:
            print(self.use_scheduled_sampling)
            sampling_thresh = tf.placeholder(tf.float32, (), 'sampling_thresh')

        self.batch_size = tf.shape(vgg_vis_features)[0]
        # self.decoder_W = tf.Variable(tf.random_uniform([self.batch_size, dim_hidden], -0.1, 0.1), name='decoder_W')

        ##===================================================
        # shallow feature representation
        # vgg_feature (dim_image) -> embeddings (dim_hidden)
        ##===================================================
        vis_features_flat = tf.reshape(vgg_vis_features, [-1, self.dim_image]) # b*n_frame, dim_img
        vis_features = tf.nn.xw_plus_b(vis_features_flat, self.vis_encode_W, self.vis_encode_b) # b*n_frame,dim_hidden
        vis_features = tf.reshape(vis_features, [self.batch_size, self.n_frames, self.dim_hidden]) #

        # state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        # state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        lstm1_memory = self.lstm1.zero_state(self.batch_size, dtype=tf.float32)
        lstm2_memory = self.lstm2.zero_state(self.batch_size, dtype=tf.float32)

        padding_lstm1 = tf.zeros([self.batch_size, self.dim_hidden]) # to replace inputs during decoding
        padding_lstm2 = tf.zeros([self.batch_size, 2*self.dim_hidden]) # attention+word_embd

        probs = []
        loss = []
        temporal_features = []
        ##############################  Encoding Stage ##################################
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, self.n_frames):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                ##====================================================
                # LSTM1
                # for each time i
                # input : embeddings (dim_hidden), long (dim_hidden)
                # output : short (dim_hidden), long (dim_hidden)
                ##====================================================
                with tf.variable_scope("LSTM1"):
                    temporal_feature, lstm1_memory = self.lstm1(vis_features[:,i,:], lstm1_memory)

                temporal_features.append(temporal_feature) # n,b,h
                ##====================================================
                # LSTM2
                # for each time i
                # input : [atn,word_embd,short] (2+1*dim_hidden), long (dim_hidden)
                # output : short (dim_hidden), long (dim_hidden)
                ##====================================================
                with tf.variable_scope("LSTM2"):
                    output2, lstm2_memory = self.lstm2(tf.concat(axis=1, values=[padding_lstm2, temporal_feature]), lstm2_memory)

            # print(type(hidden1))
            temporal_features = tf.stack(temporal_features, axis=1) # b,n,h
            # print(hidden1.shape)
            flat_temporal_features = tf.reshape(temporal_features, [-1, self.dim_hidden])  # b*n,h

        ############################# Decoding Stage ######################################
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            generated_words = []
            alphas = []
            for i in range(0, self.n_caption_lstm_step):
                # #============= the original version =============#
                # with tf.device("/cpu:0"): # suppose caption has <bos> at the beggining
                #     current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
                # #================================================#
                with tf.device("/cpu:0"):
                    if train:
                        if self.use_scheduled_sampling:
                            if i==0:
                                current_embed = tf.nn.embedding_lookup(self.word_embeddings, caption[:, i])
                            else:
                                current_embed = tf.cond(tf.random_uniform(shape=(), minval=0., maxval=1.) < sampling_thresh,
                                                        true_fn=lambda : last_embed,
                                                        false_fn=lambda : tf.nn.embedding_lookup(self.word_embeddings, caption[:, i]))

                        else :
                            current_embed = tf.nn.embedding_lookup(self.word_embeddings, caption[:, i])
                    else : # test
                        if i==0: # bos
                            current_embed = tf.nn.embedding_lookup(self.word_embeddings, tf.ones([self.batch_size], dtype=tf.int64))
                        else :
                            current_embed = last_embed

                scope.reuse_variables()

                with tf.variable_scope("LSTM1"):
                    temporal_feature, lstm1_memory = self.lstm1(padding_lstm1, lstm1_memory)

                ##=======================
                # luong attention
                # hidden1 : n,b,h -> b,n,h
                # output2 : b,h -> b,1,h
                # W : h,h
                ##=======================

                if self.luong:
                    keys = tf.nn.xw_plus_b(flat_temporal_features, self.att_W, self.att_b)#tf.layers.dense(hidden1, self.dim_hidden)
                    keys = tf.reshape(keys, [self.batch_size, -1, self.dim_hidden])
                    query = tf.expand_dims(output2, axis=1)
                    alpha = tf.matmul(query, keys, transpose_b=True)
                    alpha = tf.squeeze(alpha) # b,n
                    alpha = tf.nn.softmax(alpha)

                else:
                    ##======================#
                    # bahdaunau attention
                    # v*tanh(w1*keys+w2*query)
                    # keys : b,n_frame,h
                    # query : b,1,h
                    # v : h
                    ##======================#

                    keys = tf.reshape(tf.matmul(flat_temporal_features, self.bah_w1), (self.batch_size, self.n_frames, self.dim_hidden)) # b,n,h

                    query = tf.expand_dims(tf.matmul(output2, self.bah_w2), axis=1) # b,1,h

                    tmp = tf.nn.tanh(keys+query)

                    alpha = tf.reduce_sum(self.bah_v*tmp, axis=-1) # b,n

                    alpha = tf.nn.softmax(alpha) # b,n


                contexts = tf.multiply(tf.expand_dims(alpha, axis=-1), temporal_features)  # b,n,h
                context = tf.reduce_sum(contexts, axis=1)  # b,h

                alphas.append(alpha) # b,n_frame

                with tf.variable_scope("LSTM2"):
                    output2, lstm2_memory = self.lstm2(tf.concat(axis=1, values=[current_embed, context, temporal_feature]), lstm2_memory)


                logit_words = tf.nn.xw_plus_b(output2, self.dense_word_W, self.dense_word_b)
                max_prob_index = tf.argmax(logit_words, 1) # b,1
                generated_words.append(max_prob_index) # batch of words at time t

                if not train or self.use_scheduled_sampling:
                    with tf.device("/cpu:0"):
                        last_embed = tf.nn.embedding_lookup(self.word_embeddings, max_prob_index)
                        # last_embed = tf.expand_dims(current_embed, 0)
                        # print('last embed {}'.format(last_embed.shape))

                if train:
                    labels = tf.expand_dims(caption[:, i + 1], 1)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat(axis=1, values=[indices, labels])
                    onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                    cross_entropy = cross_entropy * caption_mask[:,i]

                    loss.append(cross_entropy)

            # we want each frame is focused approx. the same, should summed up across caps
            # n_cap * b,n_frame
            if train:
                alphas = tf.stack(alphas, axis=2) # b,n_frame,n_cap
                alphas_ = tf.reduce_sum(alphas, axis=-1) # b,n_frame

        generated_words = tf.stack(generated_words) # t,b
        generated_words = tf.transpose(generated_words) # b,t

        if train:
            loss = tf.reduce_mean(tf.stack(loss)) # t,b
            # alphas_loss = tf.reduce_mean(tf.nn.relu(alpha_baseline-alphas_))
            # loss += alph_loss_weight*alphas_loss
            summary_op = tf.summary.merge((tf.summary.histogram('alphas', alphas_),
                                           # tf.summary.scalar('alphas_loss', alphas_loss),
                                           ))
            # summary_op = tf.summary.histogram('alphas', alphas_)

        self._params_usage()

        ret = [loss, vgg_vis_features, video_mask, caption, caption_mask, generated_words]
        if train:
            ret.append(summary_op)
            if self.use_scheduled_sampling:
                ret.append(sampling_thresh)
        return ret


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_frames, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_frames])

        self.batch_size = tf.shape(video)[0]
        # self.decoder_W = tf.Variable(tf.random_uniform([self.batch_size, dim_hidden], -0.1, 0.1), name='decoder_W')

        ##===================================================
        # shallow feature representation
        # vgg_feature (dim_image) -> embeddings (dim_hidden)
        ##===================================================
        video_flat = tf.reshape(video, [-1, self.dim_image])  # b*n_frame, dim_img
        image_emb = tf.nn.xw_plus_b(video_flat, self.vis_encode_W, self.vis_encode_b)  # b*n_frame,dim_hidden
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_frames, self.dim_hidden])  #

        # state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        # state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        state1 = self.lstm1.zero_state(self.batch_size, dtype=tf.float32)
        state2 = self.lstm2.zero_state(self.batch_size, dtype=tf.float32)

        padding_lstm1 = tf.zeros([self.batch_size, self.dim_hidden])  # to replace inputs during decoding
        padding_lstm2 = tf.zeros([self.batch_size, 2 * self.dim_hidden])  # attention+word_embd



        probs = []
        embeds = []
        hidden1 = []
        ##############################  Encoding Stage ##################################
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, self.n_frames):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                ##====================================================
                # LSTM1
                # for each time i
                # input : embeddings (dim_hidden), state1 (dim_hidden)
                # output : out1 (dim_hidden), state1 (dim_hidden)
                ##====================================================
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)


                hidden1.append(output1)  # n,b,h
                ##====================================================
                # LSTM2
                # for each time i
                # input : [atn,word_embd,out1] (2+1*dim_hidden), state2 (dim_hidden)
                # output : out1 (dim_hidden), state1 (dim_hidden)
                ##====================================================
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat(axis=1, values=[padding_lstm2, output1]), state2)

            # print(type(hidden1))
            hidden1 = tf.stack(hidden1, axis=1)  # b,n,h
            # print(hidden1.shape)
            hidden1_ = tf.reshape(hidden1, [-1, self.dim_hidden])  # b*n,h

        ############################# Decoding Stage ######################################
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            generated_words = []
            alphas = []
            for i in range(0, self.n_caption_lstm_step):
                # with tf.device("/cpu:0"):
                #     current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
                tf.get_variable_scope().reuse_variables()

                if i == 0:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.word_embeddings, tf.ones([1], dtype=tf.int64))


                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding_lstm1, state1)

                ##=======================
                # luong attention
                # hidden1 : n,b,h -> b,n,h
                # output2 : b,h -> b,1,h
                # W : h,h
                ##=======================

                keys = tf.nn.xw_plus_b(hidden1_, self.att_W, self.att_b)  # tf.layers.dense(hidden1, self.dim_hidden)
                keys = tf.reshape(keys, [self.batch_size, -1, self.dim_hidden])
                query = tf.expand_dims(output2, axis=1)
                alpha = tf.matmul(query, keys, transpose_b=True)
                alpha = tf.squeeze(alpha)  # b,n
                alpha = tf.nn.softmax(alpha)
                contexts = tf.multiply(tf.expand_dims(alpha, axis=-1), hidden1)  # b,n,h
                context = tf.reduce_sum(contexts, axis=1)  # b,h

                alphas.append(tf.reshape(alpha, (self.batch_size, -1)))  # n_cap,b,n_frame


                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat(axis=1, values=[current_embed, context, output1]), state2)

                logit_words = tf.nn.xw_plus_b(output2, self.dense_word_W, self.dense_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_words.append(max_prob_index)
                probs.append(logit_words)

                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.word_embeddings, max_prob_index)
                    current_embed = tf.expand_dims(current_embed, 0)

                embeds.append(current_embed)

            alphas = tf.reduce_sum(tf.stack(alphas), axis=0)  # b,n_frame
            alphas = tf.transpose(alphas)  # n_frame, b -> for each batch, how does n_frame as a whole perform

        self._params_usage()

        return video, video_mask, generated_words, probs, embeds, alphas


def get_video_data(video_data_path, video_feat_path, train=True):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    #apply function to each row (axis=1)
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = sorted(video_data['video_path'].unique())
    data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]

    ind = int(len(data)*train_test_ratio)
    if train:
        prompts.append('with ratio {}, {} items used for training, the remaining {} for testing'.format(train_test_ratio,
                                                                                                        ind, len(data)-ind))
        return data[:ind]
    else:
        return data[ind:]

def get_video_train_data(video_data_path, video_feat_path):
    return get_video_data(video_data_path, video_feat_path, train=True)

def get_video_test_data(video_data_path, video_feat_path):
    return get_video_data(video_data_path, video_feat_path, train=False)

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    prompts.append('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        #compute word occurrence
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    prompts.append('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def word_indices_to_sentence(ixtoword, generated_word_indices):
    assert len(generated_word_indices.shape)==1
    generated_words = [ixtoword[idx] for idx in generated_word_indices]
    # print(generated_words)
    # input()

    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
    generated_words = generated_words[:punctuation]

    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace(' <eos>', '')

    return generated_sentence

def add_custom_summ(tagname, tagvalue, writer, iters):
    psnr_summ = tf.Summary()
    node = psnr_summ.value.add()
    node.tag = tagname
    node.simple_value = tagvalue

    writer.add_summary(psnr_summ, iters)

def train(restore_path=None):#os.path.join('models', '')):#os.path.join('models', '20180617_210234')):
    global prompts,learning_rate

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    prompts.append('ckpt stored in {}, summaries in {}'.format(model_path, logdir))

    ## data
    train_data = get_video_train_data(video_train_data_path, video_train_feat_path)
    train_captions = train_data['Description'].values
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_captions = test_data['Description'].values

    captions_list = list(train_captions) + list(test_captions)
    captions = np.asarray(captions_list, dtype=np.object)

    captions = [x.replace('.', '') for x in captions]
    captions = [x.replace(',', '') for x in captions]
    captions = [x.replace('"', '') for x in captions]
    captions = [x.replace('\n', '') for x in captions]
    captions = [x.replace('?', '') for x in captions]
    captions = [x.replace('!', '') for x in captions]
    captions = [x.replace('\\', '') for x in captions]
    captions = [x.replace('/', '') for x in captions]

    ## vocab
    if restore_path is None:
        wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)

        # np.save(os.path.join(model_path, 'wordtoix'), wordtoix)
        np.save(os.path.join(model_path, 'ixtoword'), ixtoword)
        np.save(os.path.join(model_path, 'bias_init_vector'), bias_init_vector)
    else :
        ixtoword = pd.Series(np.load(os.path.join(restore_path, 'ixtoword.npy')).tolist())
        wordtoix = {v: k for k, v in ixtoword.items()}
        bias_init_vector = np.load(os.path.join(restore_path, 'bias_init_vector.npy'))


    graph = tf.Graph()
    with graph.as_default():
        ## model & ops
        model = Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                scheduled_sampling=scheduled_sampling,
                # n_lstm_steps=n_frame_step,
                n_video_lstm_step=n_video_lstm_step,
                n_caption_lstm_step=n_caption_lstm_step,
                bias_init_vector=bias_init_vector)

        model_ops = model.build_model()
        tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_predicted, summary_op = model_ops[:7]
        if scheduled_sampling:
            tf_sampling = model_ops[-1]
        sess = tf.Session(graph=graph)

        # train op, init
        step = 0
        step_ = tf.Variable(0, trainable=False, name='global_step')
        advance_step = step_.assign_add(1)

        learning_rate = tf.reduce_max([tf.train.exponential_decay(learning_rate, step_, 5000, 0.9), 1e-6])
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
        summ_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())
        # fd = open('tmp1.txt', 'w')
        # for v in tf.trainable_variables():
        #     fd.write(v.name)
        #     fd.write('\n')
        #     fd.write(str(sess.run(graph.get_tensor_by_name(v.name))))
        #     fd.write('\n')
        # fd.close()

        saver = tf.train.Saver(max_to_keep=30, var_list=tf.trainable_variables())
        if restore_path is not None and not os.path.exists(restore_path):
            print('restore_model path wrong')
            exit(1)
        if restore_path is not None:
            latest_ckpt = tf.train.latest_checkpoint(restore_path)
            saver.restore(sess, latest_ckpt)
            prompts.append('restore model from {}'.format(latest_ckpt))

        # fd = open('tmp2.txt', 'w')
        # for v in tf.trainable_variables():
        #     fd.write(v.name)
        #     fd.write('\n')
        #     fd.write(str(sess.run(graph.get_tensor_by_name(v.name))))
        #     fd.write('\n')
        # fd.close()
        # input('plz check it')

        #new_saver = tf.train.Saver()
        #new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
        #new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

        loss_fd = open(os.path.join(model_path, 'loss_{}.txt'.format(time_id)), 'w')
        loss_fd.write('\n'.join(prompts))
        [print(x) for x in prompts]
        prompts.clear()
        # loss_to_draw = []

        xent = 0.

        history = []
        convertor = lambda x, y: (x * len(y) - 1) / (len(y) - 1) * 100

        if scheduled_sampling:
            # x^1/2, x=0.~1.
            sampling_threshes = [(float(x)/n_epochs)**0.5 for x in range(1, n_epochs+1)]

        for epoch in range(0, n_epochs):
            # loss_to_draw_epoch = []

            index = list(train_data.index)
            np.random.shuffle(index)
            train_data = train_data.ix[index]

            current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))]) # sample one discrp.
            current_train_data = current_train_data.reset_index(drop=True)

            total_bleu = 0.
            epoch_start = step
            for start, end in zip(
                    list(range(0, len(current_train_data), batch_size)),
                    list(range(batch_size, len(current_train_data), batch_size))):


                start_time = time.time()

                current_batch = current_train_data[start:end]
                current_videos = current_batch['video_path'].values

                current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
                current_feats_vals = [np.load(vid) for vid in current_videos]
                # print('for each in current batch :', [x.shape for x in current_feats_vals])
                # input() # to test weather the following padding is required

                current_video_masks = np.zeros((batch_size, n_video_lstm_step))

                for ind,feat in enumerate(current_feats_vals):
                    current_feats[ind][:len(current_feats_vals[ind])] = feat
                    current_video_masks[ind][:len(current_feats_vals[ind])] = 1

                current_captions = current_batch['Description'].values
                current_captions = ['<bos> ' + x for x in current_captions]
                current_captions = [x.replace('.', '') for x in current_captions]
                current_captions = [x.replace(',', '') for x in current_captions]
                current_captions = [x.replace('"', '') for x in current_captions]
                current_captions = [x.replace('\n', '') for x in current_captions]
                current_captions = [x.replace('?', '') for x in current_captions]
                current_captions = [x.replace('!', '') for x in current_captions]
                current_captions = [x.replace('\\', '') for x in current_captions]
                current_captions = [x.replace('/', '') for x in current_captions]

                ## add marks for captions
                for idx, each_cap in enumerate(current_captions):
                    word_seq = each_cap.lower().split(' ')
                    if len(word_seq) < n_caption_lstm_step:
                        current_captions[idx] = current_captions[idx] + ' <eos>'
                    else:
                        new_word = ''
                        for i in range(n_caption_lstm_step-1):
                            new_word = new_word + word_seq[i] + ' '
                        current_captions[idx] = new_word + '<eos>'

                ## turn word to index
                current_caption_ind = []
                for cap in current_captions:
                    current_word_ind = []
                    for word in cap.lower().split(' '):
                        if word in wordtoix:
                            current_word_ind.append(wordtoix[word])
                        else:
                            current_word_ind.append(wordtoix['<unk>'])
                    current_caption_ind.append(current_word_ind)

                current_caption_matrix = pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step) # b,caplen
                current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
                current_caption_masks = np.zeros_like(current_caption_matrix, dtype=int)#( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
                nonzeros = np.array( [(x != 0).sum() + 1 for x in current_caption_matrix] )

                for ind, row in enumerate(current_caption_masks): # set mask
                    row[:nonzeros[ind]] = 1

                # probs_val = sess.run(tf_probs, feed_dict={
                #     tf_video:current_feats,
                #     tf_caption: current_caption_matrix
                #     })
                ops = [train_op, advance_step, tf_loss, tf_predicted]
                if (step+1) % summ_freq == 0:
                    ops.append(summary_op)

                fd = {
                    tf_video: current_feats,
                    tf_video_mask : current_video_masks,
                    tf_caption: current_caption_matrix,
                    tf_caption_mask: current_caption_masks
                    }

                if scheduled_sampling:
                    fd[tf_sampling] = sampling_threshes[epoch]

                ret = sess.run(ops, feed_dict=fd)
                # loss_to_draw_epoch.append(loss_val)
                _, step, loss_val, predicted_captions = ret[:4]
                # step = sess.run(step_)
                xent += loss_val

                i = np.random.choice(len(predicted_captions))
                predicted_sents = [word_indices_to_sentence(ixtoword, cap) for cap in predicted_captions]
                ground_sents = [word_indices_to_sentence(ixtoword, gnd) for gnd in current_caption_matrix]
                bleus = [BLEU(predicted_sent, ground_sent) for predicted_sent,ground_sent in zip(predicted_sents,ground_sents)]
                total_bleu += np.mean(bleus)

                prompts += [
                    'epoch {}, iter {}, loss {}, elapsed {}'.format(epoch, start, loss_val, time.time() - start_time),
                    'caption #{}: bleu={}'.format(current_videos[i], bleus[i]),
                    'predicted="{}",'.format(predicted_sents[i]),
                    'groundtru="{}"'.format(ground_sents[i]),
                    '=============================']

                if step % summ_freq == 0: # after the step advance, so should be step-1
                    summ_writer.add_summary(ret[4], step)
                    tmp = xent / summ_freq
                    add_custom_summ(tagname='xent',tagvalue=tmp,writer=summ_writer,iters=step)
                    prompts += ['*****************************',
                                'in this {} steps, avg xent is {}'.format(summ_freq, tmp),
                                '*****************************']
                    xent = 0.
                # step += 1

            avg_bleu = total_bleu / (step-epoch_start)

            if epoch > save_freq:
                del history[0]
            history.append(avg_bleu)

            prompts.append(
                "\nEpoch {} with learning rate {} is done. Avg BLEU is {}".format(epoch, sess.run(learning_rate), avg_bleu))
            if scheduled_sampling:
                prompts.append('\nSampling rate for this epoch is {}'.format(sampling_threshes[epoch]))
            add_custom_summ(tagname='bleu', tagvalue=avg_bleu, writer=summ_writer, iters=epoch)
            if scheduled_sampling:
                add_custom_summ(tagname='sampling', tagvalue=sampling_threshes[epoch], writer=summ_writer, iters=epoch)

            ##===== save exceptions ======#
            thresh = 1.0 if epoch < 20 else np.percentile(history, convertor(0.8, history))

            if avg_bleu >= thresh or np.mod(epoch+1, save_freq) == 0:
                save_name = os.path.join(model_path,
                                        'loss_{}_{}'.format(
                                            str(tmp).replace('.','@'),
                                            time.strftime('%H%M%S', time.localtime())))
                saver.save(sess, save_name, global_step=epoch)

            loss_fd.write('\n'.join(prompts))
            loss_fd.write('\n')
            [print(x) for x in prompts]
            prompts.clear() # clear every epoch

        loss_fd.close()


def predict_with_restore(test_videos, restore_path=os.path.join('models', '20180618_041100'), idx=None):
    # test_data = get_video_test_data(video_test_data_path, video_test_feat_path) # video_path, Description
    # test_videos = test_data['video_path'].unique()
    wordtoix = pd.Series(np.load(os.path.join(restore_path, 'wordtoix.npy')).tolist())
    ixtoword = {v: k for k, v in wordtoix.items()}
    bias_init_vector = np.load(os.path.join(restore_path, 'bias_init_vector.npy'))


    np.save(os.path.join(restore_path, 'ixtoword'), ixtoword)



    if idx is not None:
        test_videos = test_videos[idx:idx+1]


    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            scheduled_sampling=scheduled_sampling,
            # n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_gnd_indices, tf_caption_mask, tf_predicted_indices = model.build_model(train=False)

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    if restore_path is not None and not os.path.exists(restore_path):
        print('restore_model path wrong')
        exit(1)
    if restore_path is not None:
        latest_ckpt = tf.train.latest_checkpoint(restore_path)
        saver.restore(sess, latest_ckpt)

    for video_feat_path in test_videos:

        video_feat = np.load(video_feat_path)
        # batch_size*
        video_feat = np.expand_dims(video_feat, axis=0)

        generated_word_index = sess.run(tf_predicted_indices, feed_dict={tf_video:video_feat, tf_video_mask:np.ones(video_feat.shape[:2], dtype=np.int)})

        generated_sentence = word_indices_to_sentence(ixtoword, np.squeeze(generated_word_index))
        yield generated_sentence


def test(restore_path=os.path.join('models', '20180618_041100'), idx=None):
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_videos = test_data['video_path'].unique()

    if idx is None:
        idx = input('here are {} videos for testing. Which would you like to choose?\n'.format(len(test_videos)))
        if idx == '':
            idx = np.random.choice(len(test_videos))
            print('sooooo, i randomly choose #{}:{} for you.'.format(idx, test_videos[idx]))
    print('*********************processing*********************')

    test_output_txt_fd = open(os.path.join(restore_path, 'S2VT_results.txt'), 'w')

    try:
        generated_sentence = next(predict_with_restore(test_videos=test_videos, restore_path=restore_path, idx=idx)) # this is a generator
        print('********************* caption *********************')
        print(generated_sentence, '\n')
        test_output_txt_fd.write(test_videos[idx] + '\n')
        test_output_txt_fd.write(generated_sentence + '\n\n')
    except :
        pass


def validate_all_test(restore_path=os.path.join('models', '20180619_205638')):
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_videos = test_data['video_path'].unique()
    groups = test_data.groupby('video_path')
    # print(test_data.columns)

    gen = predict_with_restore(test_videos=test_videos, restore_path=restore_path, idx=None)

    cnt = 0
    for generated_sentence in gen:
        # compute the minimum
        current_vpath = test_videos[cnt]
        cnt += 1
        current_captions = groups.get_group(current_vpath) # get all the captions
        # print(current_captions.columns)
        # input()
        bleus = [BLEU(generated_sentence, ground) for ground in current_captions['Description']]
        print(generated_sentence)
        print('average bleu score for video {} is {}'.format(current_vpath.split(os.pathsep))[-1], np.mean(bleus))
        best_index = np.argmax(bleus)
        print('the best matched ground truth with bleu={} is\n{}'.format(bleus[best_index],
                                                                         current_captions['Description'].iloc[best_index]))


if __name__ == '__main__':
    validate_all_test()