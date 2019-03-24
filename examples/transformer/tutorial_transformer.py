import tensorflow as tf
import tensorlayer as tl

from examples.transformer.models import model_params
from examples.transformer.models.transformer import Transformer
from examples.transformer.utils import wmt_dataset


def get_dataset():
    def _parse_example(serialized_example):
        """Return inputs and targets Tensors from a serialized tf.Example."""
        data_fields = {
            "inputs": tf.io.VarLenFeature(tf.int64),
            "targets": tf.io.VarLenFeature(tf.int64)
        }
        parsed = tf.io.parse_single_example(serialized_example, data_fields)
        inputs = tf.sparse.to_dense(parsed["inputs"])
        targets = tf.sparse.to_dense(parsed["targets"])
        return inputs, targets

    def _load_records(filename):
        """Read file and return a dataset of tf.Examples."""
        return tf.data.TFRecordDataset(filename, buffer_size=512)

    dataset = tf.data.Dataset.list_files('data/wmt32k-train-00001*')
    dataset = dataset.interleave(_load_records, cycle_length=2)
    dataset = dataset.map(_parse_example)

    batch_size = 1024
    max_length = 256

    dataset = dataset.padded_batch(batch_size=batch_size // max_length,
                                   padded_shapes=([max_length], [max_length]),
                                   drop_remainder=True)

    return dataset


def train_model():

    #@tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            print(inputs)
            inputs = input_layer(inputs)
            targets = target_layer(targets)
            predictions = transformer(inputs=inputs, targets=targets)
            loss = loss_object(targets, predictions, from_logits=True)
        print(loss)
        gradients = tape.gradient(loss, transformer.weights)
        optimizer.apply_gradients(zip(gradients, transformer.weights))
    #     #
    #     # train_loss(loss)
    #     # train_accuracy(label, predictions)

    params = model_params.EXAMPLE_PARAMS
    transformer = Transformer(params)

    input_layer = tl.layers.Input([None, None], dtype=tf.int32)
    target_layer = tl.layers.Input([None, None], dtype=tf.int32)
    # predictions = transformer(inputs, targets)

    loss_object = tf.losses.sparse_categorical_crossentropy
    # loss = loss_object(targets, predictions, from_logits=True)

    # model = tl.models.Model(inputs=[inputs, targets], outputs=[predictions, loss], name='transformer_model')

    optimizer = tf.optimizers.Adam()

    dataset = get_dataset()
    dataset = dataset.repeat(5)

    for inputs, targets in dataset:
        train_step(inputs, targets)
        # print(inputs.shape, targets.shape)
        # with tf.GradientTape() as tape:
        #     predictions_, loss_ = model([inputs, targets])
        # print(loss_)
        # grad = tape.gradient(loss_, model.weights)
        # optimizer.apply_gradients(zip(grad, model.weights))

    # sess = tf.InteractiveSession()
    #
    # inputs = tf.placeholder(shape=(None, None), dtype=tf.int64, name='inputs')
    # targets = tf.placeholder(shape=(None, None), dtype=tf.int64, name='targets')
    # logits = model(inputs=inputs, targets=targets)
    # loss = tf.losses.sparse_softmax_cross_entropy(targets, logits)
    # pred = tf.argmax(logits, axis=2)
    #
    # train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    #
    # iterator, next_element = get_dataset()
    # sess.run(iterator.initializer)
    # sess.run(tf.global_variables_initializer())
    # while True:
    #     try:
    #         numerical_inputs, numerical_targets = sess.run(next_element)
    #         numerical_loss, _ = sess.run([loss, train_op],
    #                                      feed_dict={inputs: numerical_inputs, targets: numerical_targets})
    #         print("loss : %" % numerical_loss)
    #     except:
    #         print("all data has been consumed: training ends")


def test_dataset():
    dataset = get_dataset()
    dataset = dataset.repeat(5)
    for input, target in dataset:
        print(input)
        break


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    #test_dataset()
    # wmt_dataset.download_and_preprocess_dataset('data/raw', 'data', search=False)
    train_model()
