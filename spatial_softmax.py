from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages
import tensorflow as tf

@add_arg_scope
def spatial_softmax(features,
                    temperature=None,
                    name=None,
                    variables_collections=None,
                    trainable=True,
                    data_format='NHWC'):
  """Computes the spatial softmax of a convolutional feature map.
  First computes the softmax over the spatial extent of each channel of a
  convolutional feature map. Then computes the expected 2D position of the
  points of maximal activation for each channel, resulting in a set of
  feature keypoints [x1, y1, ... xN, yN] for all N channels.
  Read more here:
  "Learning visual feature spaces for robotic manipulation with
  deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
  Args:
    features: A `Tensor` of size [batch_size, W, H, num_channels]; the
      convolutional feature map.
    temperature: Softmax temperature (optional). If None, a learnable
      temperature is created.
    name: A name for this operation (optional).
    variables_collections: Collections for the temperature variable.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
  Returns:
    feature_keypoints: A `Tensor` with size [batch_size, num_channels * 2];
      the expected 2D locations of each channel's feature keypoint (normalized
      to the range (-1,1)). The inner dimension is arranged as
      [x1, y1, ... xN, yN].
  Raises:
    ValueError: If unexpected data_format specified.
    ValueError: If num_channels dimension is unspecified.
  """
  with variable_scope.variable_scope(name, 'spatial_softmax'):
    shape = array_ops.shape(features)
    static_shape = features.shape
    if True:
    # if data_format == DATA_FORMAT_NHWC:
      height, width, num_channels = shape[1], shape[2], static_shape[3]
    elif data_format == DATA_FORMAT_NCHW:
      num_channels, height, width = static_shape[1], shape[2], shape[3]
    else:
      raise ValueError('data_format has to be either NCHW or NHWC.')
    if num_channels.value is None:
      raise ValueError('The num_channels dimension of the inputs to '
                       '`spatial_softmax` should be defined. Found `None`.')

    with ops.name_scope('spatial_softmax_op', 'spatial_softmax_op', [features]):
      # Create tensors for x and y coordinate values, scaled to range [-1, 1].
      pos_x, pos_y = array_ops.meshgrid(
          math_ops.lin_space(-1., 1., num=height),
          math_ops.lin_space(-1., 1., num=width),
          indexing='ij')
      pos_x = array_ops.reshape(pos_x, [height * width])
      pos_y = array_ops.reshape(pos_y, [height * width])

      if temperature is None:
        temp_initializer = init_ops.ones_initializer()
      else:
        temp_initializer = init_ops.constant_initializer(temperature)

      if not trainable:
        temp_collections = None
      else:
        temp_collections = utils.get_variable_collections(
            variables_collections, 'temperature')

      temperature = variables.model_variable(
          'temperature',
          shape=(),
          dtype=dtypes.float32,
          initializer=temp_initializer,
          collections=temp_collections,
          trainable=trainable)
      if data_format == 'NCHW':
        features = array_ops.reshape(features, [-1, height * width])
      else:
        features = array_ops.reshape(
            array_ops.transpose(features, [0, 3, 1, 2]), [-1, height * width])

      softmax_attention = nn.softmax(features / temperature)
      expected_x = math_ops.reduce_sum(
          pos_x * softmax_attention, [1], keep_dims=True)
      expected_y = math_ops.reduce_sum(
          pos_y * softmax_attention, [1], keep_dims=True)
      expected_xy = array_ops.concat([expected_x, expected_y], 1)
      feature_keypoints = array_ops.reshape(expected_xy,
                                            [-1, num_channels.value * 2])
      feature_keypoints.set_shape([None, num_channels.value * 2])
  return feature_keypoints


def stack(inputs, layer, stack_args, **kwargs):
  """Builds a stack of layers by applying layer repeatedly using stack_args.
  `stack` allows you to repeatedly apply the same operation with different
  arguments `stack_args[i]`. For each application of the layer, `stack` creates
  a new scope appended with an increasing number. For example:
  ```python
    y = stack(x, fully_connected, [32, 64, 128], scope='fc')
    # It is equivalent to:
    x = fully_connected(x, 32, scope='fc/fc_1')
    x = fully_connected(x, 64, scope='fc/fc_2')
    y = fully_connected(x, 128, scope='fc/fc_3')
  ```
  If the `scope` argument is not given in `kwargs`, it is set to
  `layer.__name__`, or `layer.func.__name__` (for `functools.partial`
  objects). If neither `__name__` nor `func.__name__` is available, the
  layers are called with `scope='stack'`.
  Args:
    inputs: A `Tensor` suitable for layer.
    layer: A layer with arguments `(inputs, *args, **kwargs)`
    stack_args: A list/tuple of parameters for each call of layer.
    **kwargs: Extra kwargs for the layer.
  Returns:
    A `Tensor` result of applying the stacked layers.
  Raises:
    ValueError: If the op is unknown or wrong.
  """
  scope = kwargs.pop('scope', None)
  if not isinstance(stack_args, (list, tuple)):
    raise ValueError('stack_args need to be a list or tuple')
  with variable_scope.variable_scope(scope, 'Stack', [inputs]):
    inputs = ops.convert_to_tensor(inputs)
    if scope is None:
      if hasattr(layer, '__name__'):
        scope = layer.__name__
      elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
        scope = layer.func.__name__  # In case layer is a functools.partial.
      else:
        scope = 'stack'
    outputs = inputs
    for i in range(len(stack_args)):
      kwargs['scope'] = scope + '_' + str(i + 1)
      layer_args = stack_args[i]
      if not isinstance(layer_args, (list, tuple)):
        layer_args = [layer_args]
      outputs = layer(outputs, *layer_args, **kwargs)
    return outputs
