# sys.path.append("/usr/local/lib/python2.7/site-packages")
import os

import cv2
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
import os
import model_RGB

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def preprocess_frame(image, target_height=224, target_width=224):
    #function to resize frames then crop
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_width,target_height))

    elif height < width:
        #cv2.resize(src, dim) , where dim=(width, height)
        #image.shape[0] returns height, image.shape[1] returns width, image.shape[2] reutrns 3 (3 RGB channels)
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_height))
        cropping_length = int((resized_image.shape[1] - target_width) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_width, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_width, target_height))


def extract_video_features(video_path, num_frames = 80, vgg16_model='./vgg16.tfmodel'):
    print("Extracting video features for: " + os.path.basename(video_path))
    # Load tensorflow VGG16 model and setup computation graph
    with open(vgg16_model, mode='rb') as f:
      fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })
    graph = tf.get_default_graph()

    # Read video file
    # print(video_path)
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        pass

    #extract frames from video
    frame_count = 0
    frame_list = []
    # print('reading')
    while True:
        #extract frames from the video, where each frame is an array (height*width*3)
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)
        frame_count += 1
    frame_list = np.array(frame_list)
    # print(len(frame_list))

    # select num_frames from frame_list if frame_cout > num_frames
    if frame_count > num_frames:
        frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
        frame_list = frame_list[frame_indices]

    # crop/resize each frame
    #cropped_frame_list is a list of frames, where each frame is a height*width*3 ndarray
    cropped_frame_list = np.asarray([preprocess_frame(x) for x in frame_list])
    # print(cropped_frame_list.shape)

    # extract fc7 features from VGG16 model for each frame
    # feats.shape = (num_frames, 4096)
    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
      video_feat = sess.run(fc7_tensor, feed_dict={images: cropped_frame_list})

    return video_feat

def get_caption(video_feat, model_path='./models/model-910'):
    print("Generating caption ...")
    #video_feat_path = os.path.join('./temp_RGB_feats', '8e0yXMa708Y_24_33.avi.npy')
    ixtoword = pd.Series(np.load('./data/ixtoword.npy', encoding='bytes').tolist())
    bias_init_vector = np.load('./data/bias_init_vector.npy', encoding='bytes')

    # lstm parameters
    dim_image = 4096
    dim_hidden= 512
    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80
    batch_size = 50

    #setup lstm encoder-decoer with attention model
    model = model_RGB.Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            # n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    #restore lstm model parameters
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    # model_path = tf.train.latest_checkpoint(model_path)
    print('restoring from {}'.format(model_path))
    saver.restore(sess, model_path)
    print('restored!')
    video_feat = video_feat[None,...]

    if video_feat.shape[1] == n_frame_step:
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
    # print(video_feat.shape)
    # input()
    # run model and obatin the embeded words (indices)
    generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})

    # convert indices to words
    generated_words = ixtoword[generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace(' <eos>', '')
    print(generated_sentence,'\n')

    return generated_sentence

if __name__ == '__main__':
    video_path = os.path.join('..','s2atn_vidcap','data','testing_data','video','0lh_UWF9ZP4_62_69.avi')
    vcap_model_path = os.path.join('upload','models','model-910') #
    vgg_model_path = os.path.join('upload','vgg16.tfmodel')
    video_feat = extract_video_features(video_path, num_frames=80, vgg16_model=vgg_model_path)
    # print(video_feat.shape)
    get_caption(video_feat, model_path=vcap_model_path)
