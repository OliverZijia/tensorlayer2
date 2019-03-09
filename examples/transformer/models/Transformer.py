import tensorlayer as tl
from examples.transformer.models import embedding_layer


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

    def forward(self, inputs):
        pass
