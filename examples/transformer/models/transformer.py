import tensorlayer as tl
import tensorflow as tf
from examples.transformer.models import embedding_layer
from examples.transformer.models.attention_layer import SelfAttentionLayer, MultiHeadAttentionLayer
from examples.transformer.models.feedforward_layer import FeedForwardLayer
from examples.transformer.models.model_utils import get_input_mask, get_target_mask, positional_encoding


# class Transformer(tl.layers.Layer):
#     """
#     Transormer model. Inherits from Layer instead of model, because we can connect
#     all layers when initializing instead of forward?
#
#     Parameters
#     ----------
#     params: a parameter object
#         refer to ../transformer/utils/model_params for details
#
#     Methods
#     ----------
#     __init__()
#         Initializing the Layer.
#     __call__()
#         (1) Building the Layer if necessary. (2) Forwarding the computation.
#     weights()
#         Return a list of Tensor which are all trainable weights of this Layer.
#     build()
#         Abstract method. Build the Layer. All trainable weights should be defined in this function.
#     forward()
#         Abstract method. Forward computation and return computation results.
#     """
#
#     def __init__(self, params):
#         super(Transformer, self).__init__()
#         self.params = params
#
#         self._nodes_fixed = True
#
#     def __repr__(self):
#         # TODO
#         return super.__repr__()
#
#     def build(self, inputs_shape):
#         self.embedding_layer = embedding_layer.EmbeddingLayer(
#             self.params.vocab_size, self.params.hidden_size)
#         self.encoder_stack = EncoderStack(self.params)
#         self.decoder_stack = DecoderStack(self.params)
#         self.dropout = tl.layers.Dropout(self.params.keep_prob)
#
#         if self._weights is None:
#             self._weights = list()
#         self._weights.extend(self.embedding_layer.weights)
#         self._weights.extend(self.encoder_stack.weights)
#         self._weights.extend(self.decoder_stack.weights)
#
#         self.output_W = self._get_weights('output_W', (self.params.hidden_size, self.params.vocab_size))
#
#     def forward(self, inputs, targets=None):
#         length = tf.shape(inputs)[1]
#         input_mask = get_input_mask(inputs)
#         target_mask = get_target_mask(length)
#
#         # inputs = tf.cast(inputs, tf.float32)
#         # inputs = self.dropout(inputs)
#         inputs = self.embedding_layer(inputs)
#         targets = self.embedding_layer(targets)
#         # shift targets to right
#         targets = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
#         inputs += positional_encoding(length, self.params.hidden_size)
#         targets += positional_encoding(length, self.params.hidden_size)
#
#         # print(inputs)
#         # print(targets)
#         # inputs = self.dropout(inputs)
#         # targets = self.dropout(targets)
#
#         features = self.encoder_stack(inputs, input_mask=input_mask)
#         outputs = self.decoder_stack(inputs=targets, features=features, input_mask=input_mask, target_mask=target_mask)
#
#         # TODO
#         # return logits
#         outputs = tf.tensordot(outputs, self.output_W, axes=[[2], [0]])
#         outputs = tf.nn.softmax(outputs)
#         return outputs


class Transformer(tl.models.Model):
    """
    Transormer model.

    Parameters
    ----------
    params: a parameter object, containing hyper-parameter values to construct model
        refer to ../transformer/utils/model_params for details

    Methods
    ----------
    __init__()
        Initializing the model, constructing all essential components
    forward()
        forward pass of the model
    """

    def __init__(self, params):
        super(Transformer, self).__init__()

        self.params = params

        self.embedding_layer = embedding_layer.EmbeddingLayer(
            params.vocab_size, params.hidden_size)
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)
        self.output_linear = OutputLinear(params)

    def forward(self, inputs, targets):
        length = tf.shape(inputs)[1]
        input_mask = get_input_mask(inputs)
        target_mask = get_target_mask(length)

        # inputs = tf.cast(inputs, tf.float32)
        # inputs = self.dropout(inputs)
        inputs = self.embedding_layer(inputs)
        targets = self.embedding_layer(targets)
        # shift targets to right
        targets = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        inputs += positional_encoding(length, self.params.hidden_size)
        targets += positional_encoding(length, self.params.hidden_size)

        # print(inputs)
        # print(targets)
        if self.is_train:
            inputs = tf.nn.dropout(inputs, rate=1 - self.params.keep_prob)
            targets = tf.nn.dropout(targets, rate=1 - self.params.keep_prob)

        features = self.encoder_stack(inputs, input_mask=input_mask)
        outputs = self.decoder_stack(inputs=targets, features=features, input_mask=input_mask, target_mask=target_mask)
        outputs = self.output_linear(outputs)
        return outputs


class OutputLinear(tl.layers.Layer):

    def __init__(self, params):
        super(OutputLinear, self).__init__()
        self.params = params

        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        self.W = self._get_weights("W", (self.params.hidden_size, self.params.vocab_size))

    def forward(self, inputs):
        inputs = tf.tensordot(inputs, self.W, axes=[[2], [0]])
        inputs = tf.nn.softmax(inputs)
        return inputs

    def __repr__(self):
        return "output linear layer"


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

        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        self.scale = self._get_weights('scale', shape=(self.hidden_size), init=tl.initializers.Ones())
        self.bias = self._get_weights('bias', shape=(self.hidden_size), init=tl.initializers.Zeros())

    def forward(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[-1], keepdims=True)
        var = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keepdims=True)
        norm_inputs = (inputs - mean) * tf.math.rsqrt(var + self.epsilon)
        return norm_inputs * self.scale + self.bias

    def __repr__(self):
        return "layer normalization"


# class SublayerWrapper(object):
#     """
#     wrapper for sublayer(attention, feedforward)
#     contains no parameters, so is not a tl layer
#     """
#
#     def __init__(self, layer, params):
#         self.layer = layer
#         self.layer_norm = LayerNormalization(params.hidden_size)
#         self.dropout = tl.layers.Dropout(keep=params.keep_prob)
#
#         self.dropout._nodes_fixed = True
#
#         self.weights = []
#         self.weights.extend(self.layer.weights)
#         self.weights.extend(self.layer_norm.weights)
#
#     def __call__(self, inputs, *args, **kwargs):
#         outputs = self.dropout(self.layer(inputs, *args, **kwargs))
#         # outputs = self.layer(inputs, *args, **kwargs)
#         # residual connection
#         return self.layer_norm(inputs + outputs)
#
#     def __repr__(self):
#         return "sublayer"


class SublayerWrapper(tl.models.Model):
    """
    wrapper for sublayer(attention, feedforward)
    contains no parameters, so is not a tl layer
    """

    def __init__(self, layer, params):
        super(SublayerWrapper, self).__init__()
        self.params = params

        self.layer = layer
        self.layer_norm = LayerNormalization(params.hidden_size)

    def forward(self, inputs, *args, **kwargs):
        outputs = self.layer(inputs, *args, **kwargs)
        if self.is_train:
            outputs = tf.nn.dropout(outputs, rate=1 - self.params.keep_prob)
        # residual connection
        return self.layer_norm(inputs + outputs)


# class EncoderStack(tl.layers.Layer):
#     """
#     Encoder stack
#     Encoder is made up of self-attn and feed forward
#
#     Parameters
#     ----------
#     params: a parameter object
#         refer to ../transformer/utils/model_params for details
#
#     """
#
#     def __init__(self, params, name):
#         super(EncoderStack, self).__init__()
#         self.params = params
#
#         self.build(tuple())
#         self._built = True
#
#     def __repr__(self):
#         return "encoder stack"
#
#     def build(self, inputs_shape):
#         self.sublayers = []
#         for _ in range(self.params.encoder_num_layers):
#             self.sublayers.append([
#                 SublayerWrapper(
#                     SelfAttentionLayer(self.params.num_heads, self.params.hidden_size,
#                                        self.params.keep_prob), self.params),
#                 SublayerWrapper(
#                     FeedForwardLayer(self.params.hidden_size, self.params.ff_size,
#                                      self.params.keep_prob), self.params)])
#         self.layer_norm = LayerNormalization(self.params.hidden_size)
#
#         # if self._weights is None:
#         #     self._weights = list()
#         # for _ in range(self.params.encoder_num_layers):
#         #     self._weights.extend(self.sublayers[_][0].weights)
#         #     self._weights.extend(self.sublayers[_][1].weights)
#         # self._weights.extend(self.layer_norm.weights)
#
#     def forward(self, inputs, input_mask):
#         """
#         Parameters
#         ----------
#         inputs:
#             inputs to the Encoder, shape=(batch_size, length, hidden_size)
#
#         Return
#         -------
#             encoded features, shape=(batch_size, length, hidden_size)
#         """
#         for sublayer in self.sublayers:
#             inputs = sublayer[0](inputs, input_mask)
#             inputs = sublayer[1](inputs)
#         inputs = self.layer_norm(inputs)
#         return inputs


class EncoderStack(tl.models.Model):
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

        self.sublayers = []
        for _ in range(params.encoder_num_layers):
            self.sublayers.append([
                SublayerWrapper(SelfAttentionLayer(params.num_heads, params.hidden_size, params.keep_prob),
                                params),
                SublayerWrapper(FeedForwardLayer(params.hidden_size, params.ff_size, params.keep_prob),
                                params)])

        self.layer_norm = LayerNormalization(params.hidden_size)

    def forward(self, inputs, input_mask):
        """
        Parameters
        ----------
        inputs:
            inputs to the Encoder, shape=(batch_size, length, hidden_size)
        input_mask:
            mask for padding

        Return
        -------
            encoded features, shape=(batch_size, length, hidden_size)
        """
        for sublayer in self.sublayers:
            inputs = sublayer[0](inputs=inputs, mask=input_mask)
            inputs = sublayer[1](inputs)
        inputs = self.layer_norm(inputs)
        return inputs


class DecoderStack(tl.models.Model):
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

        self.sublayers = []
        for _ in range(self.params.decoder_num_layers):
            self.sublayers.append([
                SublayerWrapper(
                    SelfAttentionLayer(self.params.num_heads, self.params.hidden_size,
                                       self.params.keep_prob), self.params),
                SublayerWrapper(
                    MultiHeadAttentionLayer(self.params.num_heads, self.params.hidden_size,
                                            self.params.keep_prob), self.params),
                SublayerWrapper(
                    FeedForwardLayer(self.params.hidden_size, self.params.ff_size,
                                     self.params.keep_prob), self.params)])
        self.layer_norm = LayerNormalization(self.params.hidden_size)

    def forward(self, inputs, features, input_mask, target_mask):
        for sublayer in self.sublayers:
            inputs = sublayer[0](inputs, mask=target_mask)
            inputs = sublayer[1](inputs, y=features, mask=input_mask)
            inputs = sublayer[2](inputs)
        inputs = self.layer_norm(inputs)
        return inputs

# class DecoderStack(tl.layers.Layer):
#     """
#     Decoder stack
#     Decoder is made of self-attn, src-attn, and feed forward
#
#     Parameters
#     ----------
#     params: a parameter object
#         refer to ../transformer/utils/model_params for details
#
#     """
#
#     def __init__(self, params):
#         super(DecoderStack, self).__init__()
#         self.params = params
#
#         self._nodes_fixed = True
#         if not self._built:
#             self.build(tuple())
#             self._built = True
#
#     def build(self, inputs_shape):
#         self.sublayers = []
#         for _ in range(self.params.decoder_num_layers):
#             self.sublayers.append([
#                 SublayerWrapper(
#                     SelfAttentionLayer(self.params.num_heads, self.params.hidden_size,
#                                        self.params.keep_prob), self.params),
#                 SublayerWrapper(
#                     MultiHeadAttentionLayer(self.params.num_heads, self.params.hidden_size,
#                                             self.params.keep_prob), self.params),
#                 SublayerWrapper(
#                     FeedForwardLayer(self.params.hidden_size, self.params.ff_size,
#                                      self.params.keep_prob), self.params)])
#         self.layer_norm = LayerNormalization(self.params.hidden_size)
#
#         if self._weights is None:
#             self._weights = list()
#         for _ in range(self.params.decoder_num_layers):
#             self._weights.extend(self.sublayers[_][0].weights)
#             self._weights.extend(self.sublayers[_][1].weights)
#             self._weights.extend(self.sublayers[_][2].weights)
#         self._weights.extend(self.layer_norm.weights)
#
#     def forward(self, inputs, features, input_mask, target_mask):
#         for sublayer in self.sublayers:
#             inputs = sublayer[0](inputs, target_mask)
#             inputs = sublayer[1](inputs, features, input_mask)
#             inputs = sublayer[2](inputs)
#         inputs = self.layer_norm(inputs)
#         return inputs
#
#     def __repr__(self):
#         return "decoder stack"
