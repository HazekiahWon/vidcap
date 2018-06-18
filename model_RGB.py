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
save_freq = 5
learning_rate = 0.0001
alpha_baseline = 0.3*n_caption_lstm_step / float(n_video_lstm_step)
alph_loss_weight = 0.5

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        # self.batch_size = batch_size

        self.n_frames=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=True)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.att_W = tf.Variable( tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='att_W')
        self.att_b = tf.Variable( tf.zeros([dim_hidden]), name='att_b')


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


    def build_model(self):

        video = tf.placeholder(tf.float32, [None, self.n_frames, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [None, self.n_frames])

        caption = tf.placeholder(tf.int32, [None, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [None, self.n_caption_lstm_step+1])

        self.batch_size = tf.shape(video)[0]
        # self.decoder_W = tf.Variable(tf.random_uniform([self.batch_size, dim_hidden], -0.1, 0.1), name='decoder_W')

        ##===================================================
        # shallow feature representation
        # vgg_feature (dim_image) -> embeddings (dim_hidden)
        ##===================================================
        video_flat = tf.reshape(video, [-1, self.dim_image]) # b*n_frame, dim_img
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # b*n_frame,dim_hidden
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_frames, self.dim_hidden]) #

        # state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        # state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        state1 = self.lstm1.zero_state(self.batch_size, dtype=tf.float32)
        state2 = self.lstm2.zero_state(self.batch_size, dtype=tf.float32)

        padding_lstm1 = tf.zeros([self.batch_size, self.dim_hidden]) # to replace inputs during decoding
        padding_lstm2 = tf.zeros([self.batch_size, 2*self.dim_hidden]) # attention+word_embd

        probs = []
        loss = []
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
                    output1, state1 = self.lstm1(image_emb[:,i,:], state1)

                # collecting all hidden states from LSTM1
                # if i == 0:
                #     hidden1 = tf.expand_dims(output1, axis=1) # hidden1: b * n * h
                # else:
                #     hidden1 = tf.concat([hidden1, tf.expand_dims(output1, axis=1)], axis=1)
                hidden1.append(output1) # n,b,h
                ##====================================================
                # LSTM2
                # for each time i
                # input : [atn,word_embd,out1] (2+1*dim_hidden), state2 (dim_hidden)
                # output : out1 (dim_hidden), state1 (dim_hidden)
                ##====================================================
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat(axis=1, values=[padding_lstm2, output1]), state2)

            # print(type(hidden1))
            hidden1 = tf.stack(hidden1, axis=1) # b,n,h
            # print(hidden1.shape)
            hidden1_ = tf.reshape(hidden1, [-1, self.dim_hidden])  # b*n,h

        ############################# Decoding Stage ######################################
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            generated_words = []
            alphas = []
            for i in range(0, self.n_caption_lstm_step):
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
                tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding_lstm1, state1)

                ##=======================
                # luong attention
                # hidden1 : n,b,h -> b,n,h
                # output2 : b,h -> b,1,h
                # W : h,h
                ##=======================
                #calculating the context vector

                keys = tf.nn.xw_plus_b(hidden1_, self.att_W, self.att_b)#tf.layers.dense(hidden1, self.dim_hidden)
                keys = tf.reshape(keys, [self.batch_size, -1, self.dim_hidden])
                query = tf.expand_dims(output2, axis=1)
                alpha = tf.matmul(query, keys, transpose_b=True)
                alpha = tf.squeeze(alpha) # b,n
                alpha = tf.nn.softmax(alpha)
                contexts = tf.multiply(tf.expand_dims(alpha, axis=-1), hidden1) # b,n,h
                context = tf.reduce_sum(contexts, axis=1) # b,h
                # hidden1 = tf.reshape(hidden1, [-1, self.dim_hidden]) # (b*n) * h
                # hidden2 = tf.reshape(output2, [-1, 1]) # (b*h) * 1
                # W_bilinear = tf.tile(self.att_W, tf.stack([1, batch_size])) # h * (b*h)
                # alpha = tf.matmul(hidden1, W_bilinear)
                # alpha = tf.matmul(alpha, hidden2) # (b*n) * 1
                # alpha = tf.reshape(alpha, [self.batch_size, -1]) # b * n
                # alpha = tf.nn.softmax(alpha) # b * n
                # alpha = tf.reshape(alpha, [-1,1]) # (b*n) * 1
                # context = alpha * hidden1 # (b*n) * h
                # context = tf.reshape(context, [self.batch_size, -1, self.dim_hidden]) # b * n * h
                # context = tf.reduce_sum(context, axis=1) # b * h

                alphas.append(tf.reshape(alpha, (self.batch_size,-1))) # b,n_frame


                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat(axis=1, values=[current_embed, context, output1]), state2)

                labels = tf.expand_dims(caption[:, i+1], 1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat(axis=1, values=[indices, labels])
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1) # b,1
                generated_words.append(max_prob_index) # batch of words at time t

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                cross_entropy = cross_entropy * caption_mask[:,i]

                probs.append(logit_words)

                # current_loss = tf.reduce_mean(cross_entropy)#/self.batch_size
                loss.append(cross_entropy)

            # we want each frame is focused approx. the same, should summed up across caps
            # n_cap * b,n_frame
            alphas = tf.stack(alphas, axis=2) # b,n_frame,n_cap
            alphas_ = tf.reduce_sum(alphas, axis=-1) # b,n_frame

        generated_words = tf.stack(generated_words) # t,b
        generated_words = tf.transpose(generated_words) # b,t
        loss = tf.reduce_mean(tf.stack(loss)) # t,b
        # alphas_loss = tf.reduce_mean(tf.nn.relu(alpha_baseline-alphas_))
        # loss += alph_loss_weight*alphas_loss
        summary_op = tf.summary.merge((tf.summary.histogram('alphas', alphas_),
                                       # tf.summary.scalar('alphas_loss', alphas_loss),
                                       ))
        # summary_op = tf.summary.histogram('alphas', alphas_)

        self._params_usage()

        return loss, video, video_mask, caption, caption_mask, probs, generated_words, summary_op


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
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)  # b*n_frame,dim_hidden
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
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))


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

                logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_words.append(max_prob_index)
                probs.append(logit_words)

                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
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

def train(restore_path=os.path.join('models', '')):#os.path.join('models', '20180617_210234')):
    global prompts,learning_rate
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

        np.save(os.path.join(model_path, 'wordtoix'), wordtoix)
        # np.save('./data/ixtoword', ixtoword)
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
                batch_size=batch_size,
                # n_lstm_steps=n_frame_step,
                n_video_lstm_step=n_video_lstm_step,
                n_caption_lstm_step=n_caption_lstm_step,
                bias_init_vector=bias_init_vector)

        tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs, tf_predicted, summary_op = model.build_model()
        sess = tf.Session(graph=graph)

        # train op, init
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        prompts.append('ckpt stored in {}, summaries in {}'.format(model_path, logdir))

        step = 1
        learning_rate = tf.train.exponential_decay(learning_rate, step, 5000, 0.9)
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

        saver = tf.train.Saver(max_to_keep=10, var_list=tf.trainable_variables())
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

        loss_fd = open('loss_{}.txt'.format(time_id), 'w')
        loss_fd.write('\n'.join(prompts))
        [print(x) for x in prompts]
        prompts.clear()
        # loss_to_draw = []

        xent = 0.

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
                ops = [train_op, tf_loss, tf_predicted]
                if step % summ_freq == 0:
                    ops.append(summary_op)
                ret = sess.run(
                        ops,
                        feed_dict={
                            tf_video: current_feats,
                            tf_video_mask : current_video_masks,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })
                # loss_to_draw_epoch.append(loss_val)
                _, loss_val, predicted_captions = ret[:3]
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

                if step % summ_freq == 0:
                    summ_writer.add_summary(ret[3], step)
                    tmp = xent / summ_freq
                    add_custom_summ(tagname='xent',tagvalue=tmp,writer=summ_writer,iters=step)
                    prompts += ['*****************************',
                                'in this {} steps, avg xent is {}'.format(summ_freq, tmp),
                                '*****************************']
                    xent = 0.
                step += 1

            avg_bleu = total_bleu / (step-epoch_start)

            prompts.append(
                "\nEpoch {} with learning rate {} is done. Avg BLEU is {}".format(epoch, sess.run(learning_rate), avg_bleu))
            add_custom_summ(tagname='bleu', tagvalue=avg_bleu, writer=summ_writer, iters=epoch)

            if np.mod(epoch+1, save_freq) == 0:
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


def test(restore_path=os.path.join('models', '20180618_041100'), idx=None):
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_videos = test_data['video_path'].unique()

    ixtoword = pd.Series(np.load(os.path.join(restore_path, 'ixtoword.npy')).tolist())
    # wordtoix = {v: k for k, v in ixtoword.items()}
    bias_init_vector = np.load(os.path.join(restore_path, 'bias_init_vector.npy'))

    if idx is None:
        idx = input('here are {} videos for testing. Which would you like to choose?\n'.format(len(test_videos)))
        if idx=='':
            idx = np.random.choice(len(test_videos))
            print('sooooo, i randomly choose #{}:{} for you.'.format(idx, test_videos[idx]))
    print('*********************processing*********************')


    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            # n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf, alphas = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    if restore_path is not None and not os.path.exists(restore_path):
        print('restore_model path wrong')
        exit(1)
    if restore_path is not None:
        latest_ckpt = tf.train.latest_checkpoint(restore_path)
        saver.restore(sess, latest_ckpt)

    test_output_txt_fd = open(os.path.join(restore_path, 'S2VT_results.txt'), 'w')

    video_feat_path = test_videos[idx]

    video_feat = np.load(video_feat_path)
    # batch_size*
    video_feat = np.expand_dims(video_feat, axis=0)

    generated_word_index, alphas_ = sess.run([caption_tf, alphas], feed_dict={video_tf:video_feat, video_mask_tf:np.ones(video_feat.shape[:2], dtype=np.int)})

    # atn_w = pd.DataFrame(alphas_)
    # print(atn_w.shape)
    # print(atn_w.describe())

    generated_sentence = word_indices_to_sentence(ixtoword, generated_word_index)
    print('********************* caption *********************')
    print(generated_sentence,'\n')
    test_output_txt_fd.write(video_feat_path + '\n')
    test_output_txt_fd.write(generated_sentence + '\n\n')
