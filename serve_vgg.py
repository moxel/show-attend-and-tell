from core.vggnet import Vgg19
import tensorflow as tf
import numpy as np
import json

import moxel
from moxel.space import Image, String


#img_in = Image.from_file('example/rock.jpg')
#img_in.resize((224, 224))
#
vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
vggnet = Vgg19(vgg_model_path)
vggnet.build()


def predict(ins):
    img_in = ins['image']
    img_in.resize((224, 224))
    image_batch = np.array([img_in.to_numpy()]).astype(np.float32)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
        print(feats)
    return {
        'feature': String.from_str(str(feats))
    }


