# deep-learning-project

Version 0 is the original version, with 2 convolutional layers, 2 dropout layers and 1 FC layer;

Version 1 uses higher learning rate and decay rate, in main：
learning rate: 0.001 -> 0.01
decay rate: 0.95 -> 0.9;

Version 2 applies only one dropout layer to take more information from inputs, in nets:
remove line 42 "nets = capsule_dropout(nets,drop_probability=0.25)";

Version 3 adds additional convolutional layer, in nets:
add line 44 "nets = capsules_conv(nets, kernel_shape=[3, 3], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv2',out_capsule_dims=16,out_capsule_channels=32)".
