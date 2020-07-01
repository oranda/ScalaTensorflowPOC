/* Copyright 2020, James McCabe. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package com.oranda.tensorflow

import java.nio.file.Paths

import scala.language.postfixOps

import com.oranda.tensorflow.MNISTSimple.{RawDataTensor, imageDim}

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.client.{FeedMap, Session}
import org.platanios.tensorflow.api.ops.variables.ZerosInitializer
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.data.image.MNISTLoader

/**
 * An approximate Scala translation of
 * https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_softmax.py
 *
 * Trains and tests using the tensorflow_scala API on MNIST data.
 * This is the simple version, not the CNN version.
 *
 * There is an explanation of the original Python code here:
 * https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/tutorials/mnist/pros/index.md
 *
 * @author James McCabe
 */
object MNISTSimple extends App {

  type RawDataTensor = Tensor[UByte]

  val imageDim = 784 // an image is 784 bytes, representing 28*28 pixels
  val labelDim = 10 // an image is associated with a label from 0-9, making 10 possibilities

  val trainIterations = 1000
  val trainBatchSize = 100
  val testBatchSize = 10000 // there's only 1 test iteration

  // Placeholder variables for the data
  val x_in = tf.placeholder[Float](Shape(-1, imageDim))
  val y_out = tf.placeholder[Float](Shape(-1, labelDim))

  // Training variables: weights and biases
  val w = tf.variable[Float]("W", Shape(imageDim, labelDim), ZerosInitializer)
  val b = tf.variable[Float]("b", Shape(labelDim), ZerosInitializer)

  // Define the regression model (y) used for both training and testing.
  val matmulResult = tf.matmul(x_in, w)
  val y = matmulResult + b

  // Run training and then test
  val mnist = MNISTSimple()
  mnist.train(trainIterations, trainBatchSize, x_in, y_out, y)
  val accuracy = mnist.test(testBatchSize, x_in, y_out, y)
  println(s"Accuracy: ${accuracy.scalar}")

  /** Constructs and returns a MNIST Simple class, which has the main functionality. */
  def apply(): MNISTSimple = {
    val (trainImages, trainLabels, testImages, testLabels) = MNISTIO.extractData
    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())
    new MNISTSimple(session, trainImages, trainLabels, testImages, testLabels)
  }
}

/**
 * Helper functions for interaction with the filesystem.
 */
object MNISTIO {
  private val validationSize = 5000
  private val pathToMNISTData = "datasets/MNIST"
  private val pathToModel = "model/model"

  /** Loads MNIST training and test data from the filesystem.
   *
   * The MNIST data contains 60,000 training examples and 10,000 test examples.
   * The training set includes starts with 5000 validation images and labels that must be skipped.
   * One-Hot still needs to be applied to the labels.
   *
   * @return  a tuple containing: trainImages, trainLabels, testImages, testLabels
   */
  def extractData: (RawDataTensor, RawDataTensor, RawDataTensor, RawDataTensor) = {
    val mnist = MNISTLoader.load(Paths.get(pathToMNISTData))

    // Narrow the data down to the real training sets.
    // Use +1 because for a Slice inclusive is true by default.
    val trainImages = mnist.trainImages(validationSize + 1 ::)
    val trainLabels = mnist.trainLabels(validationSize + 1 ::)

    val testImages = mnist.testImages
    val testLabels = mnist.testLabels

    (trainImages, trainLabels, testImages, testLabels)
  }

  /** Save a checkpoint to the filesystem. Use during/after training. */
  def save(session: Session): Unit = {
    val saver = tf.Saver()
    saver.save(session, Paths.get(pathToModel))
  }
}

/** Main class manages a TensorFlow session and allows training, testing, and saving. */
class MNISTSimple(
  session: Session,
  trainImages: RawDataTensor,
  trainLabels: RawDataTensor,
  testImages: RawDataTensor,
  testLabels: RawDataTensor
) {
  type ImageTensor = Tensor[Float]
  type LabelTensor = Tensor[Float]
  type Accuracy = Tensor[Float]

  /** Trains the model. Up to 1000 iterations are supported, and a batch size of 100. */
  def train(
    iterations: Int,
    batchSize: Int,
    x_in: Output[Float],
    y_out: Output[Float],
    y: Output[Float]
  ): Unit = {
    // For training use GradientDescent, the simplest optimizer
    val gradientDescentTrainStep = {
      val crossEntropy = -tf.sum(y_out * tf.log(ops.NN.softmax(logits=y)))
      // In Python, the learning rate is 0.5f
      //  - but that causes NaNs in the tensorflow_scala implementation.
      tf.train.GradientDescent(0.005f).minimize(crossEntropy)
    }

    // TODO: shuffling the batches gives a slight performance improvement
    for (i <- 0 until iterations) {
      if (i % 100 == 0)
        println(s"Processing batch $i of $iterations")

      val from = (i * batchSize) % (trainImages.shape(0) - batchSize)
      val feedMap = makeFeedMap(trainImages, trainLabels, batchSize, from, x_in, y_out)
      session.run(feeds = feedMap, targets = gradientDescentTrainStep)
    }

    MNISTIO.save(session)
  }

  /** Do prediction using the model. Supports a batch size of 100. */
  def test(
    batchSize: Int,
    x_in: Output[Float],
    y_out: Output[Float],
    y: Output[Float]
  ): Accuracy = {
    def correctPrediction = tf.equal(
      ops.Math.argmax(input = y, axes = 1, outputDataType = INT64),
      ops.Math.argmax(input = y_out, axes = 1, outputDataType = INT64)
    )
    val accuracy = ops.Math.mean(correctPrediction.castTo[Float])

    val testFeedMap = makeFeedMap(testImages, testLabels, batchSize, 0, x_in, y_out)
    session.run(feeds = testFeedMap, fetches = accuracy)
  }

  /** Create a feed map. This is input to training and testing. */
  private def makeFeedMap(
    rawImages: RawDataTensor,
    rawLabels: RawDataTensor,
    batchSize: Int,
    from: Int,
    x_in: Output[Float],
    y_out: Output[Float]
  ): FeedMap = {
    val to = from + batchSize

    def normalizeImagesBatch(imagesBatch: RawDataTensor, batchSize: Int): ImageTensor =
      imagesBatch.reshape(Shape(batchSize, imageDim)).castTo(FLOAT32) / 255

    def normalizeLabelsBatch(labelsBatch: RawDataTensor, batchSize: Int): Output[Float] =
      labelsBatch.oneHot[Int](10).castTo(FLOAT32).reshape(Shape(batchSize, 10))

    val imagesBatch: ImageTensor = normalizeImagesBatch(rawImages(from :: to), batchSize)
    val labelsBatch: Output[Float] = normalizeLabelsBatch(rawLabels(from :: to), batchSize)

    // The labels batch has been converted to an Output[Float] by oneHot
    //  - but in order to be in the training FeedMap, it needs to evaluated to a Tensor
    val labelsBatchTensor: LabelTensor = session.run(fetches = labelsBatch)

    FeedMap(Map(x_in -> imagesBatch, y_out -> labelsBatchTensor))
  }
}