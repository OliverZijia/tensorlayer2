import tensorflow as tf

_NEG_INF = -1e9


def get_input_mask(inputs, padding_value=0):
    """
    Parameters
    ----------
    inputs:
        input sequence, shape=(batch_size, length)
    padding_value:
        the value of padding in dictionary

    Return
    ----------
    mask:
        shape=(batch_size, 1, 1, length)
    """
    mask = tf.cast(tf.equal(inputs, padding_value), dtype=float)
    mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
    mask *= _NEG_INF
    return mask


def get_target_mask(length):
    """
    Parameters
    ----------
    length:
        length of target sequence

    Return
    ----------
    mask:
        shape=(1, 1, length, length)
    """
    # upper triangle
    mask = tf.linalg.band_part(tf.ones((length, length)), -1, 0)
    # lower triangle without diagonal
    mask = 1 - mask
    mask = tf.reshape(mask, (1, 1, length, length))
    mask *= _NEG_INF
    return mask


def positional_encoding(length, hidden_size):
    """
    Parameters
    ----------
    length:
        length of target sequence
    hidden_size:
        embedding hidden size

    Return
    ----------
    pe:
        shape=(1, length, hidden_size)
    """
    #pe = tf.zeros((length, hidden_size))
    position = tf.cast(tf.expand_dims(tf.range(length), axis=1), dtype=tf.float32)
    #div_term = tf.exp(-1 * tf.cast(tf.range(0, hidden_size, 2), dtype=tf.float32) * (tf.math.log(10000) / hidden_size))
    div_term = -1 * tf.range(0, hidden_size, 2, dtype=tf.float32)
    div_term = div_term * tf.math.log(10000.0)
    div_term = div_term / hidden_size
    div_term = tf.exp(div_term)

    # pe[:, 0::2] = tf.sin(position * div_term)
    pe = tf.concat([tf.sin(position * div_term), tf.cos(position * div_term)], axis=1)
    pe = tf.expand_dims(pe, axis=0)
    return pe

