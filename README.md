# Weight-Share
Demonstration of using Weight Share for training deep CNN's.

Deep learning has been popularized through its ability to achieve better than human performance in tasks like vision. However, in order to train these deep neural nets (typically CNN's), several data sets has been collected and merged into one. In the merging process, the images are resized to the same shape.
In many fields, e.g. spectroscopy, this resizing cannot be done without disturbing their interpretation. 

Weigth Share is a novel method for training deep CNN's on multiple data set without resizing each data set to have the same input size.

The demonstration is based on the paper "Deep learning for Chemometric and non-translational data by Larsen,J.S. and Clemmensen, L. (2019)". https://arxiv.org/abs/1910.00391

The code is developed using Python 3.6 and Tensorflow 1.12
