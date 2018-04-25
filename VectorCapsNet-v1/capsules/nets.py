"""An implementation of matrix capsules with EM routing.
"""

import tensorflow as tf

from .core import _conv2d_wrapper, capsules_init,capsules_init_v2, capsules_conv,capsules_fc,flatten_conv_capsule,flatten_dense_capsule,capsule_length,capsule_dropout

slim = tf.contrib.slim

# ------------------------------------------------------------------------------#
# -------------------------------- capsules net --------------------------------#
# ------------------------------------------------------------------------------#

def capsules_v0(inputs,labels, num_classes, iterations, name='CapsuleEM-V0'):
  """Replicate the network in `Matrix Capsules with EM Routing.`
  """

  with tf.variable_scope(name) as scope:

    # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
    #nets = _conv2d_wrapper(inputs, shape=[5, 5, 1, 8], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv1')
    # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32x(4x4+1) -> (poses, activations)
    nets = capsules_init_v2(
      inputs, shape=[9, 9, 1, 32], strides=[1, 2, 2, 1], padding='SAME', output_capsule_dims=8, name='capsule_init'
    )

    # --------------------------------- #
    # another version of initial caps layer
    #nets1 = _conv2d_wrapper(
    #  inputs, shape=[5, 5, 1, 8], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu,
    #  name='conv1'
    #)
    # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32x(4x4+1) -> (poses, activations)
    #nets = capsules_init(
    #  nets1, shape=[1, 1, 8, 8], strides=[1, 1, 1, 1], padding='VALID', output_capsule_dims=8, name='capsule_init'
    #)
    # ---------------------------------- #

    #nets = capsule_dropout(nets,drop_probability=0.25)
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 2 -> (poses, activations)
    nets = capsules_conv(nets, kernel_shape=[5, 5], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1',out_capsule_dims=16,out_capsule_channels=16)
    nets = capsule_dropout(nets,drop_probability=0.25)
    # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 1 -> (poses, activations)
    #nets = capsules_conv(nets, kernel_shape=[3, 3], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv2',out_capsule_dims=16,out_capsule_channels=32)
    #nets = capsule_dropout(nets,drop_probability=0.25)
    nets = capsules_conv(nets, kernel_shape=[3, 3], strides=[1, 1, 1, 1], iterations=iterations, name='capsule_conv3',out_capsule_dims=16, out_capsule_channels=16)
    #nets = capsule_dropout(nets, drop_probability=0.25)
    nets2 = flatten_conv_capsule(nets)
    nets = capsule_dropout(nets2,drop_probability=0.25)
    # inputs: (poses, activations) -> capsule-fc 1x1x32x10x4x4 shared view transform matrix within each channel -> (poses, activations)
    #nets = capsules_fc( nets, iterations=iterations, name='capsule_fc',out_capsule_dims=16,out_capsule_number=20)
    #nets = capsule_dropout(nets,drop_probability=0.25)
    nets = capsules_fc(nets, iterations=iterations, name='capsule_fc2', out_capsule_dims=16, out_capsule_number=10)
    #nets6 = capsules_fc(nets6, iterations=iterations, name='capsule_fc2', out_capsule_dims=16, out_capsule_number=10)
                        
    #nets = flatten_dense_capsule(nets)

    #activations = tf.layers.dense(nets,units=10,activation=tf.nn.relu)

    activation = capsule_length(nets)
    #nets = flatten_dense_capsule(nets)
    #activations = tf.layers.dense(nets,units=10,activation=tf.nn.relu)
    #activations = nets
    #activations = tf.nn.softmax(nets7,axis=-1)
    #activations = tf.layers.dense(nets,units=10,activation=None)

    mask = tf.expand_dims(labels,-1)
    mask = tf.tile(mask,[1,1,16])
    mask = tf.cast(mask,tf.float32)

    nets_recon = tf.multiply(mask,nets)
    nets_recon = tf.reduce_sum(nets_recon,axis=-2)

    #nets_recon = tf.layers.dense(nets_recon,units = 512,activation=tf.nn.relu)
    #nets_recon = tf.layers.dense(nets_recon,units = 1024,activation=tf.nn.relu)
    #nets_recon = tf.layers.dense(nets_recon,units = 28*28,activation=tf.nn.relu)
    #nets_recon = tf.reshape(nets_recon,shape=[-1,28,28,1])
  return activation,nets_recon

# ------------------------------------------------------------------------------#
# ------------------------------------ loss ------------------------------------#
# ------------------------------------------------------------------------------#

def spread_loss(labels, activations, margin, name):
  """This adds spread loss to total loss.

  :param labels: [N, O], where O is number of output classes, one hot vector, tf.uint8.
  :param activations: [N, O], activations.
  :param margin: margin 0.2 - 0.9 fixed schedule during training.

  :return: spread loss
  """

  activations_shape = activations.get_shape().as_list()

  with tf.variable_scope(name) as scope:

    mask_t = tf.equal(labels, 1)
    mask_i = tf.equal(labels, 0)

    activations_t = tf.reshape(
      tf.boolean_mask(activations, mask_t), [activations_shape[0], 1]
    )
    activations_i = tf.reshape(
      tf.boolean_mask(activations, mask_i), [activations_shape[0], activations_shape[1] - 1]
    )

    # margin = tf.Print(
    #   margin, [margin], 'margin', summarize=20
    # )

    gap_mit = tf.reduce_sum(
      tf.square(
        tf.nn.relu(
          margin - (activations_t - activations_i)
        )
      )
    )

    # tf.add_to_collection(
    #   tf.GraphKeys.LOSSES, gap_mit
    # )
    #
    # total_loss = tf.add_n(
    #   tf.get_collection(
    #     tf.GraphKeys.LOSSES
    #   ), name='total_loss'
    # )

    tf.losses.add_loss(gap_mit)

    return gap_mit

# ------------------------------------------------------------------------------#

