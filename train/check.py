# @Author:      HgS_1217_
# @Create Date: 2018/6/23

import tensorflow as tf
from config import CKPT_PREFIX
from data_process.data_preprocessor import generate_patch, ImagePreprocessor


def get_data(dataset_path):
    image_pre = ImagePreprocessor(base_dir=dataset_path)
    _, _, d = image_pre.get_dataset_patched(size=20, data_selection='all', label_type='int', exist='new')
    return d


def main(dataset_path):
    xs = get_data(dataset_path)

    ckpt = tf.train.get_checkpoint_state(CKPT_PREFIX)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('input_x').outputs[0]
    y = tf.get_collection('prediction')[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)

        results = []

        for i in range(len(xs)):
            pred = sess.run(y, feed_dict={x: xs[i], keep_prob: 1.0, is_training: True})
            results.append(pred)

    print(results)


if __name__ == '__main__':
    # generate_patch()
    main("D:/Computer Science/dataset/HPA_ieee_test")
