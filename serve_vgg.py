from core.vggnet import Vgg19
import tensorflow as tf
import numpy as np
import json

import moxel
from moxel.space import Image, String, Array


vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
vggnet = Vgg19(vgg_model_path)
vggnet.build()


def predict(image):
    image.resize((224, 224))
    image_batch = np.array([image.to_numpy()]).astype(np.float32)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
    return {
        # 'feature': String.from_str(str(feats))
        'feature': Array.from_numpy(feats)
    }


