import os
import json
import tensorflow as tf
slim = tf.contrib.slim

def binary_crossentropy(labels, predictions):
    losses = -labels * tf.log(predictions) - (1 - labels) * tf.log(1 - predictions)
    losses = tf.reduce_sum(tf.reduce_mean(losses, axis = 1), name = 'binary_crossentropy')

    return losses

def get_dataset(split_name, dataset_dir, partition_id, partition_num, reader=None):
    assert(split_name in ['train', 'val', 'test'])

    os.chdir(dataset_dir)
    with open("num_samples.json", 'r') as f:
        num_samples = json.load(f)

    with open("label_map.json", 'r') as f:
        labels_to_names = json.load(f)

    num_classes = len(labels_to_names.keys())

    file_path = os.path.join(dataset_dir,
                    '{}_{:03}-of-{:03}.tfrecord'.format(split_name, partition_id, partition_num))

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature([num_classes], tf.int64),
        #'image/class/label': tf.VarLenFeature(tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    items_to_descriptions = {
        'image': 'A color image of varying size.',
        'label': 'A single integer between 0 and {}'.format(num_classes - 1),
    }

    return slim.dataset.Dataset(
            data_sources=file_path,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=num_samples[split_name][str(partition_id-1)],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
