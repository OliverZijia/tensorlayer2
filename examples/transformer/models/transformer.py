import tensorlayer as tl
import tensorflow as tf
from examples.transformer.models import embedding_layer
from examples.transformer.models.attention_layer import SelfAttentionLayer, MultiHeadAttentionLayer
from examples.transformer.models.feedforward_layer import FeedForwardLayer
from examples.transformer.models.model_utils import get_input_mask, get_target_mask, positional_encoding


class Transformer(tl.layers.Layer):
    """
    Transormer model. Inherits from Layer instead of model, because we can connect
    all layers when initializing instead of forward?

    Parameters
    ----------
    params: a parameter object
        refer to ../transformer/utils/model_params for details

    Methods
    ----------
    __init__()
        Initializing the Layer.
    __call__()
        (1) Building the Layer if necessary. (2) Forwarding the computation.
    weights()
        Return a list of Tensor which are all trainable weights of this Layer.
    build()
        Abstract method. Build the Layer. All trainable weights should be defined in this function.
    forward()
        Abstract method. Forward computation and return computation results.
    """

    def __init__(self, params):
        super(Transformer, self).__init__(name='transformer')
        self.params = params

    def __repr__(self):
        # TODO
        return super.__repr__()

    def build(self, inputs_shape):
        self.embedding_layer = embedding_layer.EmbeddingLayer(
            self.params.vocab_size, self.params.hidden_size)
        self.encoder_stack = EncoderStack(self.params)
        self.decoder_stack = DecoderStack(self.params)
        self.dropout = tl.layers.Dropout(self.params.keep_prob)

    def forward(self, inputs, targets=None):
        length = tf.shape(inputs)[1]
        input_mask = get_input_mask(inputs)
        target_mask = get_target_mask(length)

        inputs = self.embedding_layer(inputs)
        targets = self.embedding_layer(targets)
        # shift targets to right
        targets = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        inputs += positional_encoding(length, self.params.hidden_size)
        targets += positional_encoding(length, self.params.hidden_size)

        # print(inputs)
        # print(targets)
        inputs = self.dropout(inputs)
        targets = self.dropout(targets)

        features = self.encoder_stack(inputs, input_mask=input_mask)
        outputs = self.decoder_stack(features, decoder_inputs=targets, input_mask=input_mask, target_mask=target_mask)

        # TODO
        # return logits
        return outputs


class LayerNormalization(tl.layers.Layer):
    """
    Layer normalization

    Parameters
    ----------
    hidden_size:
        hidden size of features
    epsilon:
        value to prevent division by zero

    """

    def __init__(self, hidden_size, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon

    def build(self, inputs_shape):
        self.scale = self._get_weights('scale', shape=(self.hidden_size), init=tl.initializers.ones)
        self.bias = self._get_weights('bias', shape=(self.hidden_size), init=tl.initializers.zeros)

    def forward(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[-1], keep_dims=True)
        var = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keep_dims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(var + self.epsilon)
        return norm_inputs * self.scale + self.bias

    def __repr__(self):
        pass


class SublayerWrapper(object):
    """
    wrapper for sublayer(attention, feedforward)
    contains no parameters, so is not a tl layer
    """

    def __init__(self, layer, params):
        self.layer = layer
        self.layer_norm = LayerNormalization(params.hidden_size)
        self.dropout = tl.layers.Dropout(keep=params.keep_prob)

    def __call__(self, inputs):
        outputs = self.dropout(self.layer(inputs))
        # residual connection
        return self.layer_norm(inputs + outputs)


class EncoderStack(tl.layers.Layer):
    """
    Encoder stack
    Encoder is made up of self-attn and feed forward

    Parameters
    ----------
    params: a parameter object
        refer to ../transformer/utils/model_params for details

    """

    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params

    def __repr__(self):
        pass

    def build(self, inputs_shape):
        self.sublayers = []
        for _ in range(self.params.encoder_num_layers):
            self.sublayers.append([
                SublayerWrapper(
                    SelfAttentionLayer(self.params.num_heads,
                                       self.params.hidden_size, self.params.keep_prob)),
                SublayerWrapper(
                    FeedForwardLayer(self.params.hidden_size,
                                     self.params.ff_size, self.params.keep_prob))])
        self.layer_norm = LayerNormalization(self.params.hidden_size)

    def forward(self, inputs, input_mask):
        """
        Parameters
        ----------
        inputs:
            inputs to the Encoder, shape=(batch_size, length, hidden_size)

        Return
        -------
            encoded features, shape=(batch_size, length, hidden_size)
        """
        for sublayer in self.sublayers:
            inputs = sublayer[0](inputs, input_mask)
            inputs = sublayer[1](inputs)
        inputs = self.layer_norm(inputs)
        return inputs


class DecoderStack(tl.layers.Layer):
    """
    Decoder stack
    Decoder is made of self-attn, src-attn, and feed forward

    Parameters
    ----------
    params: a parameter object
        refer to ../transformer/utils/model_params for details

    """

    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params

    def build(self, inputs_shape):
        self.sublayers = []
        for _ in range(self.params.decoder_num_layers):
            self.sublayers.append([
                SublayerWrapper(
                    SelfAttentionLayer(self.params.num_heads,
                                       self.params.hidden_size, self.params.keep_prob)),
                SublayerWrapper(
                    MultiHeadAttentionLayer(self.params.num_heads,
                                            self.params.hidden_size, self.params.keep_prob)),
                SublayerWrapper(
                    FeedForwardLayer(self.params.hidden_size,
                                     self.params.ff_size, self.params.keep_prob))])
        self.layer_norm = LayerNormalization(self.params.hidden_size)

    def forward(self, decoder_inputs, features, input_mask, target_mask):
        for sublayer in self.sublayers:
            outputs = sublayer[0](decoder_inputs, target_mask)
            outputs = sublayer[1](outputs, features, input_mask)
            outputs = sublayer[2](outputs)
        outputs = self.layer_norm(outputs)
        return outputs

    def __repr__(self):
        pass
