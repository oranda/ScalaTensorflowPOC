This project demonstrates how Scala can be used effectively with the ML library TensorFlow.

TensorFlow is usually used with Python. They also provide a Java API but it is not very powerful 
so this project uses the excellent [tensorflow_scala project](https://eaplatanios.github.io/tensorflow_scala)
from Emmanouil Antonios Platanios.

The code here is an approximate Scala translation of 
[MNIST Python code](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_softmax.py).
It trains and tests using the tensorflow_scala API on MNIST data. For a full explanation of the original Python 
code [go here](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/tutorials/mnist/pros/index.md).
I wrote a simple version for this project, not a CNN version like the one provided by Platanios.

This main code of this project is easy to browse: [MNISTSimple.scala](src/main/scala/com/oranda/tensorflow/MNISTSimple.scala).

Installation
============

1. Clone or pull this project to a convenient folder on your local machine. MacOS or Ubuntu should work.

2. This project works with TensorFlow 1.x. If you have not set up TensorFlow before, 
you will need the `libtensorflow.so` and `libtensorflow_framework.so` dynamic libraries. For convenience, 
I have provided copies in the `libs` directory. You need to ensure both those libraries are in your `LD_LIBRARY_PATH`. 
For example, on a Mac you might move them to your `/usr/local/lib/` directory and have this in your .bash_profile:	
  `export LD_LIBRARY_PATH=/usr/local/lib`  

3. If you are working on a modern Mac, you will probably run into errors unless you disable SIP. There is some
conflict between SIP and TensorFlow.
[Here](https://mc.ai/training-your-neural-net-with-egpu-acceleration-on-mac-with-tensorflow-1-5/) are some instructions.

4. You should have Scala installed. This code has been tested with Scala 2.12.8 and sbt 1.x.

Usage
=====

In the directory where you installed this project, run:

```shell
sbt run
```

After compilation you should see the main output. Like this:

```shell
2020-07-01 16:39:10.133 [run-main-0] INFO  MNIST Data Loader - Extracting images from file 'datasets/MNIST/train-images-idx3-ubyte.gz'.
2020-07-01 16:39:10.591 [run-main-0] INFO  MNIST Data Loader - Extracting labels from file 'datasets/MNIST/train-labels-idx1-ubyte.gz'.
2020-07-01 16:39:10.598 [run-main-0] INFO  MNIST Data Loader - Extracting images from file 'datasets/MNIST/t10k-images-idx3-ubyte.gz'.
2020-07-01 16:39:10.659 [run-main-0] INFO  MNIST Data Loader - Extracting labels from file 'datasets/MNIST/t10k-labels-idx1-ubyte.gz'.
2020-07-01 16:39:10.661 [run-main-0] INFO  MNIST Data Loader - Finished loading the MNIST dataset.
Processing batch 0 of 1000
Processing batch 100 of 1000
Processing batch 200 of 1000
Processing batch 300 of 1000
Processing batch 400 of 1000
Processing batch 500 of 1000
Processing batch 600 of 1000
Processing batch 700 of 1000
Processing batch 800 of 1000
Processing batch 900 of 1000
2020-07-01 16:42:01.428 [run-main-0] INFO  Variables / Saver - Saving parameters to '/Users/james/dev/ScalaTensorflowPOC/model/model'.
2020-07-01 16:42:02.080 [run-main-0] INFO  Variables / Saver - Saved parameters to '/Users/james/dev/ScalaTensorflowPOC/model/model'.
Accuracy: 0.9156
```




 