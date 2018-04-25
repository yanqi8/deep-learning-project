"""An implementation of capsule network.
"""

from tensorflow.contrib.layers.python.layers import initializers

import tensorflow as tf

import numpy as np
slim = tf.contrib.slim

epsilon = 1e-9
# ------------------------------------------------------------------------------#
#                    operations with dynamic routing in capsule nets
# ------------------------------------------------------------------------------#
def weight_variable(shape, stddev=0.1, verbose=False):
  """Creates a CPU variable with normal initialization. Adds summaries.

  Args:
    shape: list, the shape of the variable.
    stddev: scalar, standard deviation for the initilizer.
    verbose: if set add histograms.

  Returns:
    Weight variable tensor of shape=shape.
  """
  with tf.device('/cpu:0'):
    with tf.name_scope('weights'):
      weights = tf.get_variable(
          'weights',
          shape,
          initializer=tf.truncated_normal_initializer(
              stddev=stddev, dtype=tf.float32),
          dtype=tf.float32)
  #variable_summaries(weights, verbose)
  return weights


def bias_variable(shape, verbose=False):
  """Creates a CPU variable with constant initialization. Adds summaries.

  Args:
    shape: list, the shape of the variable.
    verbose: if set add histograms.

  Returns:
    Bias variable tensor with shape=shape.
  """
  with tf.device('/cpu:0'):
    with tf.name_scope('biases'):
      biases = tf.get_variable(
          'biases',
          shape,
          initializer=tf.constant_initializer(0.1),
          dtype=tf.float32)
  #variable_summaries(biases, verbose)
  return biases

import keras.backend as K
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = tf.reduce_sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1.0 + s_squared_norm) / tf.sqrt(s_squared_norm + epsilon)
    #return tf.nn.softmax(vectors,axis)
    return scale * vectors
def extract_max_capsule(inputs):
    return tf.reduce_max(inputs,axis=-1)
def capsule_length(inputs):
    """
    :param inputs.shape = [N,num_capsule,capsule_dim]
        or inputs.shape = [N,H,W,num_capsule,capsule_dime]
    :return shape =[N,num_capsule]
        or  shape = [N,H,W,num_capsule]
    """
    return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))


def _leaky_routing(logits, output_dim):
  """Adds extra dimmension to routing logits.

  This enables active capsules to be routed to the extra dim if they are not a
  good fit for any of the capsules in layer above.

  Args:
    logits: The original logits. shape is
      [input_capsule_num, output_capsule_num] if fully connected. Otherwise, it
      has two more dimmensions.
    output_dim: The number of units in the second dimmension of logits.

  Returns:
    Routing probabilities for each pair of capsules. Same shape as logits.
  """

  # leak is a zero matrix with same shape as logits except dim(2) = 1 because
  # of the reduce_sum.
  leak = tf.zeros_like(logits, optimize=True)
  leak = tf.reduce_sum(leak, axis=2, keep_dims=True)
  leaky_logits = tf.concat([leak, logits], axis=2)
  leaky_routing = tf.nn.softmax(leaky_logits, dim=2)
  return tf.split(leaky_routing, [1, output_dim], 2)[1]


def _update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                    num_routing=1, leaky=None):
  # if not routing
  # return squash(tf.reduce_sum(votes,axis=1))
  """Sums over scaled votes and applies squash to compute the activations.

  Iteratively updates routing logits (scales) based on the similarity between
  the activation of this layer and the votes of the layer below.

  Args:
    votes: tensor, The transformed outputs of the layer below.
    biases: tensor, Bias variable.
    logit_shape: tensor, shape of the logit to be initialized.
    num_dims: scalar, number of dimmensions in votes. For fully connected
      capsule it is 4, for convolutional 6.
    input_dim: scalar, number of capsules in the input layer.
    output_dim: scalar, number of capsules in the output layer.
    num_routing: scalar, Number of routing iterations.
    leaky: boolean, if set use leaky routing.

  Returns:
    The activation tensor of the output layer after num_routing iterations.
  """
  votes_t_shape = [3, 0, 1, 2]
  for i in range(num_dims - 4):
    votes_t_shape += [i + 4]
  r_t_shape = [1, 2, 3, 0]
  for i in range(num_dims - 4):
    r_t_shape += [i + 4]
  votes_trans = tf.transpose(votes, votes_t_shape)
  
  # if stop gradient
  #tf.stop_gradient(votes_trans)

  def _body(i, logits, activations):
    """Routing while loop."""
    # if stop gradient
    #tf.stop_gradient(logits)
    #tf.stop_gradient(activations)
    # route: [batch, input_dim, output_dim, ...]
    if leaky:
      route = _leaky_routing(logits, output_dim)
    else:
      route = tf.nn.softmax(logits, dim=2)
    preactivate_unrolled = route * votes_trans
    preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
    preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
    activation = squash(preactivate)
    activations = activations.write(i, activation)
    # distances: [batch, input_dim, output_dim]
    act_3d = tf.expand_dims(activation, 1)
    tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
    tile_shape[1] = input_dim
    act_replicated = tf.tile(act_3d, tile_shape)
    distances = tf.reduce_sum(votes * act_replicated, axis=3)
    logits += distances
    return (i + 1, logits, activations)

  activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False)
  logits = tf.fill(logit_shape, 0.0)

  # if stop gradient
  tf.stop_gradient(logits)

  i = tf.constant(0, dtype=tf.int32)
  _, logits, activations = tf.while_loop(
      lambda i, logits, activations: i < num_routing,
      _body,
      loop_vars=[i, logits, activations],
      swap_memory=True)

  return activations.read(num_routing - 1)


def dynamic_routing(inputs, routings, W, bias,num_dims=4):

  # return squash(tf.tensordot(inputs, W, axes=[[1, 2], [3, 1]]))
  # inputs.shape=[None, input_num_capsule, input_dim_capsule]
  # W is a tensor with shape [num_capsule,input_num_capsule,dim_capsule,input_dim_capsule]
  input_tensor = inputs
  input_dim = W.shape[1]
  output_dim = W.shape[0]
  layer_name = 'dynamic_routing'
  input_atoms = W.shape[3]
  output_atoms = W.shape[2]
  W = tf.reshape(W,shape=[input_dim,input_atoms,output_atoms*output_dim])
  """Builds a fully connected capsule layer.

  Given an input tensor of shape `[batch, input_dim, input_atoms]`, this op
  performs the following:

    1. For each input capsule, multiples it with the weight variable to get
      votes of shape `[batch, input_dim, output_dim, output_atoms]`.
    2. Scales the votes for each output capsule by iterative routing.
    3. Squashes the output of each capsule to have norm less than one.

  Each capsule of this layer has one weight tensor for each capsules of layer
  below. Therefore, this layer has the following number of trainable variables:
    w: [input_dim * num_in_atoms, output_dim * num_out_atoms]
    b: [output_dim * num_out_atoms]

  Args:
    input_tensor: tensor, activation output of the layer below.
    input_dim: scalar, number of capsules in the layer below.
    output_dim: scalar, number of capsules in this layer.
    layer_name: string, Name of this layer.
    input_atoms: scalar, number of units in each capsule of input layer.
    output_atoms: scalar, number of units in each capsule of output layer.
    **routing_args: dictionary {leaky, num_routing}, args for routing function.

  Returns:
    Tensor of activations for this layer of shape
      `[batch, output_dim, output_atoms]`.
  """
  with tf.variable_scope(layer_name):
    # weights variable will hold the state of the weights for the layer
    weights = W#variables.weight_variable(
        #[input_dim, input_atoms, output_dim * output_atoms])
    biases = bias#ariables.bias_variable([output_dim, output_atoms])
    with tf.name_scope('Wx_plus_b'):
      # Depthwise matmul: [b, d, c] ** [d, c, o_c] = [b, d, o_c]
      # To do this: tile input, do element-wise multiplication and reduce
      # sum over input_atoms dimmension.
      input_tiled = tf.tile(
          tf.expand_dims(input_tensor, -1),
          [1, 1, 1, output_dim * output_atoms])
      votes = tf.reduce_sum(input_tiled * weights, axis=2)
      votes_reshaped = tf.reshape(votes,
                                  [-1, input_dim, output_dim, output_atoms])
    with tf.name_scope('routing'):
      input_shape = tf.shape(input_tensor)
      logit_shape = tf.stack([input_shape[0], input_dim, output_dim])
      activations = _update_routing(
          votes=votes_reshaped,
          biases=biases,
          logit_shape=logit_shape,
          num_dims=num_dims,
          input_dim=input_dim,
          output_dim=output_dim,
          num_routing=routings)
    return activations

# ------------------------------------------------------------------------------#
# ------------------------------------ init ------------------------------------#
# ------------------------------------------------------------------------------#
def capsule_dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = tf.less(tf.random_uniform(shape=X.shape[1:-1],minval=0.0,maxval=1.0) , keep_probability)
    mask = tf.cast(mask,tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.expand_dims(mask,0)
    if tf.rank(X) == 3:
        mask = tf.tile(mask,[X.shape[0],1,X.shape[-1]])
    elif tf.rank(X) ==5:
        mask = tf.tile(mask,[X.shape[0],1,1,1,X.shape[-1]])
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale
def _matmul_broadcast(x, y, name):
  """Compute x @ y, broadcasting over the first `N - 2` ranks.
  """
  with tf.variable_scope(name) as scope:
    return tf.reduce_sum(
      x[..., tf.newaxis] * y[..., tf.newaxis, :, :], axis=-2
    )


def _get_variable_wrapper(
  name, shape=None, dtype=None, initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None
):
  """Wrapper over tf.get_variable().
  """

  with tf.device('/cpu:0'):
    var = tf.get_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, trainable=trainable,
      collections=collections, caching_device=caching_device,
      partitioner=partitioner, validate_shape=validate_shape,
      custom_getter=custom_getter
    )
  return var


def _get_weights_wrapper(
  name, shape, dtype=tf.float32, initializer=initializers.xavier_initializer(),
  weights_decay_factor=None
):
  """Wrapper over _get_variable_wrapper() to get weights, with weights decay factor in loss.
  """

  weights = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  if weights_decay_factor is not None and weights_decay_factor > 0.0:

    weights_wd = tf.multiply(
      tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss'
    )

    tf.add_to_collection('losses', weights_wd)

  return weights


def _get_biases_wrapper(
  name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0)
):
  """Wrapper over _get_variable_wrapper() to get bias.
  """

  biases = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  return biases

# ------------------------------------------------------------------------------#
# ------------------------------------ main ------------------------------------#
# ------------------------------------------------------------------------------#

def _conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.conv2d().
  """

  with tf.variable_scope(name) as scope:
    kernel = _get_weights_wrapper(
      name='weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.conv2d(
      inputs, filter=kernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _separable_conv2d_wrapper(inputs, depthwise_shape, pointwise_shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.separable_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=depthwise_shape, weights_decay_factor=0.0
    )
    pkernel = _get_weights_wrapper(
      name='pointwise_weights', shape=pointwise_shape, weights_decay_factor=0.0
    )
    output = tf.nn.separable_conv2d(
      input=inputs, depthwise_filter=dkernel, pointwise_filter=pkernel,
      strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      biases = _get_biases_wrapper(
        name='biases', shape=[pointwise_shape[-1]]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output


def _depthwise_conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
  """Wrapper over tf.nn.depthwise_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(
      name='depthwise_weights', shape=shape, weights_decay_factor=0.0
    )
    output = tf.nn.depthwise_conv2d(
      inputs, filter=dkernel, strides=strides, padding=padding, name='conv'
    )
    if add_bias:
      d_ = output.get_shape()[-1].value
      biases = _get_biases_wrapper(
        name='biases', shape=[d_]
      )
      output = tf.add(
        output, biases, name='biasAdd'
      )
    if activation_fn is not None:
      output = activation_fn(
        output, name='activation'
      )

  return output

# ------------------------------------------------------------------------------#
# ---------------------------------- capsules ----------------------------------#
# ------------------------------------------------------------------------------#

def capsules_init(inputs, shape, strides, padding, output_capsule_dims, name):
  """This constructs a primary capsule layer from a regular convolution layer.

  :param inputs: a regular convolution layer with shape [N, H, W, C],
    where often N is batch_size, H is height, W is width, and C is channel.
  :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
    where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
  :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param padding: padding, often SAME or VALID.
  :param output_capsule_dims: the length of output vector, PH
    where PH is pose height, and PW is pose width.
  :param name: name.

  :return:
    poses: [N, H, W, C, PH],
    where often N is batch_size, H is output height, W is output width, C is output channels,
    and PH is vector length

  note: with respect to the paper, matrix capsules with EM routing, figure 1,
    this function provides the operation to build from
    ReLU Conv1 [batch_size, 14, 14, A] to
    PrimaryCapsule poses [batch_size, 14, 14, B, 4, 4], activations [batch_size, 14, 14, B] with
    Kernel [A, B, 4 x 4 + 1], specifically,
    weight kernel shape [1, 1, A, B], strides [1, 1, 1, 1], output_capsule_dims [4, 4]
  """

  # assert len(output_capsule_dims) == 2

  with tf.variable_scope(name) as scope:

    # poses: simplified build all at once
    poses = _conv2d_wrapper(
      inputs,
      shape=shape[0:-1] + [shape[-1] * output_capsule_dims ],
      strides=strides,
      padding=padding,
      add_bias=False,
      activation_fn=tf.nn.relu,
      name='pose_stacked'
    )
    poses_shape = poses.get_shape().as_list()
    poses = tf.reshape(
      poses, shape=[-1] + poses_shape[1:-1] + [shape[-1], output_capsule_dims], name='poses'
    )

  return squash(poses)

def capsules_init_v2(inputs, shape, strides, padding, output_capsule_dims, name,initial_atoms=256,iterations=1):
  """This constructs a primary capsule layer from a regular convolution layer.

  :param inputs: a regular convolution layer with shape [N, H, W, C],
    where often N is batch_size, H is height, W is width, and C is channel.
  :param shape: the shape of convolution operation kernel, [KH, KW, I, O],
    where KH is kernel height, KW is kernel width, I is inputs channels, and O is output channels.
  :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param padding: padding, often SAME or VALID.
  :param output_capsule_dims: the length of output vector, PH
    where PH is pose height, and PW is pose width.
  :param name: name.

  :return:
    poses: [N, H, W, C, PH],
    where often N is batch_size, H is output height, W is output width, C is output channels,
    and PH is vector length

  note: with respect to the paper, matrix capsules with EM routing, figure 1,
    this function provides the operation to build from
    ReLU Conv1 [batch_size, 14, 14, A] to
    PrimaryCapsule poses [batch_size, 14, 14, B, 4, 4], activations [batch_size, 14, 14, B] with
    Kernel [A, B, 4 x 4 + 1], specifically,
    weight kernel shape [1, 1, A, B], strides [1, 1, 1, 1], output_capsule_dims [4, 4]
  """

  # assert len(output_capsule_dims) == 2

  with tf.variable_scope(name) as scope:
    # build a conv layer, output has shape [N,H,W,initial_atoms]
    conv1 = _conv2d_wrapper(
      inputs,
      shape=shape[0:-1] + [initial_atoms ],
      strides=strides,
      padding=padding,
      add_bias=False,
      activation_fn=tf.nn.relu,
      name='pose_stacked'
    )
    # conv1.shape = [N,H,W,initial_atoms]
    conv1 = tf.expand_dims(conv1,axis=-2)
    # conv1. shape = [N,H,W,1,initial_atoms]

    poses = capsules_conv(conv1, shape[0:2],strides, iterations, 'initial_conv',output_capsule_dims,shape[-1],padding='SAME')



  return poses


def _depthwise_conv3d(input_tensor,
                      kernel,
                      input_dim,
                      output_dim,
                      input_atoms=8,
                      output_atoms=8,
                      stride=2,
                      padding='SAME'):
  """Performs 2D convolution given a 5D input tensor.

  This layer given an input tensor of shape
  `[batch, input_dim, input_atoms, input_height, input_width]` squeezes the
  first two dimmensions to get a 4D tensor as the input of tf.nn.conv2d. Then
  splits the first dimmension and the last dimmension and returns the 6D
  convolution output.

  Args:
    input_tensor: tensor, of rank 5. Last two dimmensions representing height
      and width position grid.
    kernel: Tensor, convolutional kernel variables.
    input_dim: scalar, number of capsules in the layer below.
    output_dim: scalar, number of capsules in this layer.
    input_atoms: scalar, number of units in each capsule of input layer.
    output_atoms: scalar, number of units in each capsule of output layer.
    stride: scalar, stride of the convolutional kernel.
    padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels.

  Returns:
    6D Tensor output of a 2D convolution with shape
      `[batch, input_dim, output_dim, output_atoms, out_height, out_width]`,
      the convolution output shape and the input shape.
      If padding is 'SAME', out_height = in_height and out_width = in_width.
      Otherwise, height and width is adjusted with same rules as 'VALID' in
      tf.nn.conv2d.
  """
  with tf.name_scope('conv'):
    input_shape = tf.shape(input_tensor)
    _, _, _, in_height, in_width = input_tensor.get_shape()
    # Reshape input_tensor to 4D by merging first two dimmensions.
    # tf.nn.conv2d only accepts 4D tensors.

    input_tensor_reshaped = tf.reshape(input_tensor, [
        input_shape[0] * input_dim, input_atoms, input_shape[3], input_shape[4]
    ])
    input_tensor_reshaped.set_shape((None, input_atoms, in_height.value,
                                     in_width.value))
    conv = tf.nn.conv2d(
        tf.transpose(input_tensor_reshaped,[0,2,3,1]),
        kernel,
        [1, stride, stride,1],
        padding=padding,
        data_format='NHWC')
    conv = tf.transpose(conv,[0,3,1,2])
    conv_shape = tf.shape(conv)
    _, _, conv_height, conv_width = conv.get_shape()
    # Reshape back to 6D by splitting first dimmension to batch and input_dim
    # and splitting second dimmension to output_dim and output_atoms.

    conv_reshaped = tf.reshape(conv, [
        input_shape[0], input_dim, output_dim, output_atoms, conv_shape[2],
        conv_shape[3]
    ])
    conv_reshaped.set_shape((None, input_dim, output_dim, output_atoms,
                             conv_height.value, conv_width.value))
    return conv_reshaped, conv_shape, input_shape

def capsules_conv(inputs, kernel_shape,strides, iterations, name,out_capsule_dims,out_capsule_channels,padding='SAME'):
  """This constructs a convolution capsule layer from a primary or convolution capsule layer.

  :param inputs: a primary or convolution capsule layer with poses and activations,
    poses shape [N, H, W, C, PH]
  :param kernel_shape: the shape of convolution operation kernel, [KH, KW],
    where KH is kernel height, KW is kernel width
  :param strides: strides [1, SH, SW, 1] w.r.t [N, H, W, C], often [1, 1, 1, 1], or [1, 2, 2, 1].
  :param iterations: number of iterations in Dynamic routing, often 3.
  :param name: name.
  :param out_capsule_dims: dimension of output capsule
  :param out_capsule_channels: channels of output capsules

  :return: poses: [N, H, W, C_out, PH_out]

  """
  '''
  def conv_slim_capsule(input_tensor,
                      input_dim,
                      output_dim,
                      layer_name,
                      input_atoms=8,
                      output_atoms=8,
                      stride=2,
                      kernel_size=5,
                      padding='SAME',
                      **routing_args):
  '''
  input_tensor = tf.transpose(inputs,[0,3,4,1,2])
  input_dim = inputs.shape[3]
  output_dim = out_capsule_channels
  layer_name = name
  input_atoms = inputs.shape[4]
  output_atoms = out_capsule_dims
  stride = strides[1] # strides = [1,3,2,1] e.g
  with tf.variable_scope(layer_name):
    kernel = weight_variable(shape=[
      kernel_shape[0], kernel_shape[1], input_atoms, output_dim * output_atoms
    ])
    biases = bias_variable([output_dim, output_atoms, 1, 1])
    votes, votes_shape, input_shape = _depthwise_conv3d(
      input_tensor, kernel, input_dim, output_dim, input_atoms, output_atoms,
      stride, padding)

    with tf.name_scope('routing'):
      logit_shape = tf.stack([
        input_shape[0], input_dim, output_dim, votes_shape[2], votes_shape[3]
      ])
      biases_replicated = tf.tile(biases,
                                  [1, 1, votes_shape[2], votes_shape[3]])
      activations = _update_routing(
        votes=votes,
        biases=biases_replicated,
        logit_shape=logit_shape,
        num_dims=6,
        input_dim=input_dim,
        output_dim=output_dim,
        num_routing=iterations,
      leaky=None)
    return tf.transpose(activations,[0,3,4,1,2])

def flatten_conv_capsule(inputs):
    """
    :param inputs is output from a convolutional capsule layer
           inputs.shape = [N,OH,OW,C,PH] C is channel number, PH is vector length
    :return shape = [N,OH*OW*C,PH]
    """
    inputs_shape = inputs.shape
    l=[]
    for i1 in range(inputs_shape[1]):
      for i2 in range(inputs_shape[2]):
        for i3 in range(inputs_shape[3]):
          l.append(inputs[:,i1,i2,i3,:])

    out = tf.stack(l,axis=1)
    return out

def capsules_fc(inputs, iterations, name,out_capsule_dims,out_capsule_number):
  """This constructs an output layer from a primary or convolution capsule layer via
    a full-connected operation with one view transformation kernel matrix shared across each channel.

  :param inputs: a primary layer
    inputs shape [N, input_number_capsule, PH]
  :param iterations: number of iterations in EM routing, often 3.
  :param name: name.
  :param out_capsule_number, number of output capsule
  :param out_capsule_dims, dimension of output capsule

  :return: shape=[N,CO,PO] where CO is channel of output, PO is length of output vector

  """
  #inputs = tf.reshape(inputs,shape=[-1,inputs.shape[1]*inputs.shape[2],inputs.shape[-1]])

  with tf.variable_scope(name) as scope:
    W = tf.get_variable("DenseWeights",shape=[out_capsule_number,inputs.shape[1],out_capsule_dims,inputs.shape[2]],initializer=initializers.xavier_initializer())
    bias = tf.get_variable("DenseBias",shape=[out_capsule_number,out_capsule_dims],initializer=initializers.xavier_initializer())
    out = dynamic_routing(inputs, iterations, W,bias,num_dims=4)
    return out#+tf.tile(bias,[out.shape[0],1,1])

def flatten_dense_capsule(inputs):
  """
  :param inputs.shape = [N,O,PH] O is number of capusle, PH is dimension of capsule
  :return shape = [N,O*PH]
  """
  return tf.reshape(inputs,shape=[-1,inputs.shape[1]*inputs.shape[2]])


