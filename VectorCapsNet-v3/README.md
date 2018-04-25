# VectorCapsNet
This project implements convolutional capsule net, based on [Hinton's paper](https://arxiv.org/abs/1710.09829), <br> 
The implementation is based on other implementations, including [capsuleEM](https://github.com/gyang274/capsulesEM), [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras), and [tensorflow](https://github.com/Sarasra/models). <br>
<br>
It provides an easy interface for capsnet. <br>
# Library functions include: <br>
1. create_init_capsule (create_init_capsule_v2), input is a 2D image with mutli channels, output is a 2D capsule layer, output shaoe is [None,Height, Width,capsule_channel,capsule_dim], where capsule_channel is number of capsules, capsule_dim is the length of each vector.<br>
2. capsule_conv, input is of shape [None, Height, widtion, capsule_channel, capsule_dim], output shape is [None, Height2, widtion2, capsule_channel2, capsule_dim2], performs dynamic routing in a convolutional way. <br>
3. capsule_fc, fully connected capsule layer<br>
4. capsule_length, calculate length of each vector.<br>
5. flatten_conv_capsules, reshape [None, Height, Widtion, channel, dim] into [None, capsule_number,dim]<br>
6. flatten_capsule, reshape [None,capsule_number, capsule_dim] into [None,-1]<br>
<br>
The structure of network will be updated later. <br>
<br>
Requirements:<br>
python >= 3.6, tensorflow, keras <br>
<br>
Usage:<br>

```
$ python main.py
```

# Network Structure<br>
Network structure is defined in /capsules/nets.py <br>
```
def capsules_v0(inputs,labels, num_classes, iterations, name='CapsuleEM-V0'):
  """Replicate the network in `Matrix Capsules with EM Routing.`
  """

  with tf.variable_scope(name) as scope:

    # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
    
    nets = capsules_init_v2(
      inputs, shape=[9, 9, 1, 32], strides=[1, 2, 2, 1], padding='SAME', output_capsule_dims=8, name='capsule_init'
    )

    
    nets = capsules_conv(nets, kernel_shape=[5, 5], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1',out_capsule_dims=16,out_capsule_channels=16)
    nets = capsule_dropout(nets,drop_probability=0.25)
   
    nets = capsules_conv(nets, kernel_shape=[3, 3], strides=[1, 1, 1, 1], iterations=iterations, name='capsule_conv3',out_capsule_dims=16, out_capsule_channels=16)
   
    nets2 = flatten_conv_capsule(nets)
    nets = capsule_dropout(nets2,drop_probability=0.25)
    
    nets = capsules_fc(nets, iterations=iterations, name='capsule_fc2', out_capsule_dims=16, out_capsule_number=10)
    

    activation = capsule_length(nets)
   
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
  ```
# Output <br>
Three files recording training,validation and test error, <br>
minibatch_error.txt <br>
validation_error.txt <br>
test_error.txt <br>

# To do:<br> 
<del>1. Include routing algorithm </del><br>
2. Extend the idea of dense net.
