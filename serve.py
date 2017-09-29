from core.vggnet import Vgg19
import tensorflow as tf
import numpy as np
import json
import cPickle as pickle

import moxel
from moxel.space import Image, String, Array


vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
vggnet = Vgg19(vgg_model_path)
vggnet.build()

model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                   dim_hidden=1500, n_time_step=16, prev2out=True,
                                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

sess_vgg = tf.Session()
sess_cap = tf.Session()
sess_vgg.run(tf.initialize_all_variables())

with open('./data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)


def predict(ins):
    img_in = ins['image']
    img_in.resize((224, 224))
    image_batch = np.array([img_in.to_numpy()]).astype(np.float32)

    feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})

    alphas, betas, sampled_captions = model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, self.test_model)
        features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
        feed_dict = { self.model.features: features_batch }
        alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
        decoded = decode_captions(sam_cap, self.model.idx_to_word)
    return {
        # 'feature': String.from_str(str(feats))
        'feature': Array.from_numpy(feats)
    }


