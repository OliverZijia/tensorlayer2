import tensorflow as tf

from examples.transformer.utils import wmt_dataset


def get_dataset():
    def _parse_example(serialized_example):
        """Return inputs and targets Tensors from a serialized tf.Example."""
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64)
        }
        parsed = tf.parse_single_example(serialized_example, data_fields)
        inputs = tf.sparse_tensor_to_dense(parsed["inputs"])
        targets = tf.sparse_tensor_to_dense(parsed["targets"])
        return inputs, targets

    def _load_records(filename):
        """Read file and return a dataset of tf.Examples."""
        return tf.data.TFRecordDataset(filename, buffer_size=512)

    dataset = tf.data.Dataset.list_files('data/wmt32k-train-*')
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _load_records, cycle_length=2))
    dataset = dataset.map(_parse_example)

    batch_size = 1024
    max_length = 256

    dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(
        batch_size // max_length, ([max_length], [max_length])))

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return iterator, next_element


def train_model():
    params = model_params.EXAMPLE_PARAMS
    model = Transformer(params)

    sess = tf.InteractiveSession()

    inputs = tf.placeholder(shape=(None, None), dtype=tf.int64, name='inputs')
    targets = tf.placeholder(shape=(None, None), dtype=tf.int64, name='targets')
    logits = model(inputs, targets)
    loss = tf.losses.sparse_softmax_cross_entropy(targets, logits)
    pred = tf.argmax(logits, axis=2)

    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    iterator, next_element = get_dataset()
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    while True:
        try:
            numerical_inputs, numerical_targets = sess.run(next_element)
            numerical_loss, _ = sess.run([loss, train_op],
                                         feed_dict={inputs: numerical_inputs, targets: numerical_targets})
            print("loss : %" % numerical_loss)
        except:
            print("all data has been consumed: training ends")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    wmt_dataset.download_and_preprocess_dataset('data/raw', 'data', search=False)
    train_model()
