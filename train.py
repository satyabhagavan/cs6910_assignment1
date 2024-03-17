# importing packages
import argparse
import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#Activation functions
def relu(x):
  return np.maximum(0, x)

def sigmoid(x):
  clip_x = np.clip(x, -500, 500)  # Clipping x to avoid overflow
  return 1 / (1 + np.exp(-clip_x))

def _tanh(x):
  clip_x = np.clip(x, -500, 500)  # Clipping x for uniformity
  return np.tanh(clip_x)

def identity(x):
  return x

## Neural Networks class
class NeuralNetwork:
  def __init__(self, inputSize, hiddenLayers, outputSize, sizeOfHiddenLayers, batchSize, 
               learningRate, initialisationType, optimiser, epochs, activationFunc, weightDecay, 
               isWandb = False, lossFunc = "cross_entropy", dataset = "fashion_mnist", 
               betha1 = 0.9, betha2 = 0.999, betha = 0.9, epsilon = 1e-8):
    # initialising model parameters
    nodes_in_layers = []
    for i in range(hiddenLayers):
      nodes_in_layers.append(sizeOfHiddenLayers)
    nodes_in_layers.append(outputSize)
    if dataset == "fashion_mnist":
      (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
      X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    elif dataset == "mnist":
      (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
      X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)



    # normalsing and resisizing all the images
    X_train = X_train/255.0
    X_test  = X_test/255.0
    X_val   = X_val/255.0

    X_train = X_train.reshape(X_train.shape[0], 784).T
    X_test = X_test.reshape(X_test.shape[0], 784).T
    X_val = X_val.reshape(X_val.shape[0], 784).T

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_val   = X_val
    self.Y_val   = Y_val
    self.X_test  = X_test
    self.Y_test  = Y_test

    self.inputSize = inputSize
    self.outputSize= outputSize
    self.batchSize = batchSize
    self.layers = hiddenLayers + 1
    self.nodes  = nodes_in_layers
    self.initialisationType = initialisationType
    self.betha1 = betha1
    self.betha2 = betha2
    self.betha  = betha
    self.epsilon= epsilon
    self.Weights= {}
    self.Baises = {}
    self.optimiser = optimiser
    self.epochs = epochs
    self.learningRate = learningRate
    self.activationFunc = activationFunc
    self.isWandb = isWandb
    self.weightDecay = weightDecay
    self.lossFunc = lossFunc # "cross_entropy" or "MSE"
    self.dataset = dataset

  def Initialise(self):
    # initialising weights and biases as a key value pair
    W = {}
    B = {}

    PreActivation = {}
    Activation = {}

    # adding input layer
    LayerWise = self.nodes
    LayerWise.insert(0, self.inputSize)

    # initialisation of weights and baises
    for i in range(self.layers):
      if self.initialisationType == "random":
        W[i+1] = 0.01*np.random.randn(LayerWise[i+1], LayerWise[i])
        B[i+1] = 0.01*np.random.randn(LayerWise[i+1], 1)
      if self.initialisationType == "xavier":
        W[i+1] = np.random.randn(LayerWise[i+1], LayerWise[i]) * np.sqrt(2. / (LayerWise[i] + LayerWise[i+1]))
        B[i+1] = np.zeros((LayerWise[i+1], 1))

      # preactivation and activation will have same size
      PreActivation[i+1] = np.zeros((LayerWise[i+1], 1))
      Activation[i+1]    = np.zeros((LayerWise[i+1], 1))

    del LayerWise[0]

    self.Weights = W
    self.Baises  = B
    self.PreActivation = PreActivation
    self.Activation = Activation

    return W,B,PreActivation,Activation

  def InitialiseEmptyWeightsAndBiases(self):
    W = {}
    B = {}
    LayerWise = self.nodes

    LayerWise.insert(0, self.inputSize)
    for i in range(self.layers):
      W[i+1] = np.zeros((LayerWise[i+1], LayerWise[i]))
      B[i+1] = np.zeros((LayerWise[i+1], 1))
    del LayerWise[0]

    return W,B

  def FeedForward(self, x, W, B, preActivation, activation):
    # no of layers
    n = len(W)
    y = x
    for i in range(1, n+1):
      preActivation[i] = np.dot(W[i], y) + B[i]
      if self.activationFunc == "sigmoid":
        activation[i] = sigmoid(preActivation[i])
      elif self.activationFunc == "tanh":
        activation[i] = _tanh(preActivation[i])
      elif self.activationFunc == "relu":
        activation[i] = relu(preActivation[i])
      elif self.activationFunc == "identity":
        activation[i] = identity(preActivation[i])


      y = activation[i]

    # last layer we don't need activation
    y = preActivation[n]
    # doing softmax doing the each column wise
    exp_y = np.exp(y - np.max(y, axis=0, keepdims=True))  # Improve numerical stability
    y = exp_y / np.sum(exp_y, axis=0, keepdims=True)
    return y

  def BackWardPropogation(self, X, y_corr, W, preActivation, activation, y_hat):
    # y_hat is the prediction and y_corr is the correct class
    dw = {}
    db = {}

    # these many points are there in the batch
    batch_Size = y_corr.shape[0]
    y = np.zeros([10, batch_Size])
    # this y is encoded in 10*batchsize with each col being for one point that will be one

    for ind in range(batch_Size):
      y[y_corr[ind]][ind] = 1

    if self.lossFunc == "cross_entropy":
      da = y_hat - y
    else:
      da = (y_hat - y)*(y_hat)*(1 - y_hat)
      da = da / self.batchSize

    activation[0] = X
    layer = len(W)
    dh = da #used for finding next layer
    while layer >= 1:
      dw[layer] = np.dot(da, activation[layer-1].T)
      db[layer] = da.sum(axis=1, keepdims=True)
      if layer > 1:
        dh = np.dot(W[layer].T, da)
        if self.activationFunc == "sigmoid":
          dg = activation[layer-1] * (1 - activation[layer-1])
        elif self.activationFunc == "tanh":
          dg = (1 + activation[layer-1]) * (1 - activation[layer-1])
        elif self.activationFunc == "relu":
          dg = np.where(preActivation[layer-1] > 0, 1, 0)
        elif self.activationFunc == "identity":
          dg = np.ones_like(preActivation[layer-1])
          

        # dg = activation[layer-1] * (1 - activation[layer-1])
        da = dh * dg
        # hedamant product
      layer -= 1

    # L2 regularisation
    for i in range(1, self.layers+1):
      dw[i] = dw[i] + self.weightDecay*W[i]

    return dw, db

  def FindAccuracyAndLoss(self, W, B, data, labels):
    n = data.shape[1]
    correct = 0
    labels_one_hot = np.eye(10)[labels]
    #running the data on the weights and baises
    y = data
    for i in range(1, self.layers+1):
      preActivation = np.dot(W[i], y) + B[i]

      if self.activationFunc == "sigmoid":
        activation = sigmoid(preActivation)
      elif self.activationFunc == "tanh":
        activation = _tanh(preActivation)
      elif self.activationFunc == "relu":
        activation = relu(preActivation)
      elif self.activationFunc == "identity":
        activation = identity(preActivation)
      y = activation

    # last layer we don't need activation
    y = preActivation
    # doing softmax doing the each column wise
    exp_y = np.exp(y - np.max(y, axis=0, keepdims=True))  # Improve numerical stability
    y = exp_y / np.sum(exp_y, axis=0, keepdims=True)
    loss = 0

    for i in range(1, 1+self.layers):
      loss += self.weightDecay * np.linalg.norm(W[i])

    for i in range(n):
      y_pred = np.argmax(y[:,i])
      if labels[i] == y_pred:
        correct += 1

      if self.lossFunc == "cross_entropy":
        loss += -1*np.log(y[:,i][labels[i]] + 1e-9)
      else:
        loss += np.sum((y[:, i] - labels_one_hot[i]) ** 2)

    return (correct*100/ n), (loss/n)

  def predict(self, data):
    n = data.shape[1]
    y = data

    for i in range(1, self.layers+1):
      preActivation = np.dot(self.Weights[i], y) + self.Baises[i]

      if self.activationFunc == "sigmoid":
        activation = sigmoid(preActivation)
      elif self.activationFunc == "tanh":
        activation = _tanh(preActivation)
      elif self.activationFunc == "relu":
        activation = relu(preActivation)
      elif self.activationFunc == "identity":
        activation = identity(preActivation)
      y = activation

    # last layer we don't need activation
    y = preActivation
    # doing softmax doing the each column wise
    exp_y = np.exp(y - np.max(y, axis=0, keepdims=True))  # Improve numerical stability
    y = exp_y / np.sum(exp_y, axis=0, keepdims=True)
    predictions = []
    for i in range(n):
      y_pred = np.argmax(y[:,i])
      predictions.append(y_pred)

    return predictions

  def SGD(self):
    W, B, preActivation, activation  = self.Initialise()
    iteration = 0
    layers = self.layers
    empty_W, empty_B = self.InitialiseEmptyWeightsAndBiases()

    while(iteration < self.epochs):
      i = 0
      while i < self.X_train.shape[1]:
        y = self.FeedForward(self.X_train[:, i:i+self.batchSize], W, B, preActivation, activation)
        # these are the partial derivates for one point
        dw, db = self.BackWardPropogation(self.X_train[:, i:i+self.batchSize], self.Y_train[i:i+self.batchSize], W, preActivation, activation, y)

        # we will update the weights now
        for k in range(1, layers+1):
            W[k] = W[k] - self.learningRate*dw[k]
            B[k] = B[k] - self.learningRate*db[k]

        i += self.batchSize
      acuu, loss = self.FindAccuracyAndLoss(W, B, self.X_train, self.Y_train)
      v_acc, v_loss = self.FindAccuracyAndLoss(W, B, self.X_val, self.Y_val)
      if self.isWandb == True:
        wandb.log({'accuracy': acuu})
        wandb.log({'loss': loss})
        wandb.log({'v_accuracy': v_acc})
        wandb.log({'v_loss': v_loss})
        wandb.log({'epoch': iteration})
      print(acuu, loss, v_acc, v_loss)
      iteration += 1

    self.Weights = W
    self.Baises  = B

  def MomentBasedGradientDecent(self):
    W, B, preActivation, activation  = self.Initialise()
    iteration = 0
    u_W, u_B = self.InitialiseEmptyWeightsAndBiases()
    # inititialising u to be zero

    while(iteration < self.epochs):
      i = 0
      while i < self.X_train.shape[1]:
        # batch wise forward and backward passes
        y = self.FeedForward(self.X_train[:, i:i+self.batchSize], W, B, preActivation, activation)
        dw, db = self.BackWardPropogation(self.X_train[:, i:i+self.batchSize], self.Y_train[i:i+self.batchSize], W, preActivation, activation, y)

        # update the momentum with the gradient
        for k in range(1, self.layers+1):
          u_W[k] = u_W[k]*self.betha + dw[k]
          u_B[k] = u_B[k]*self.betha + db[k]

        # we will update the weights now with the momentum
        for k in range(1, self.layers+1):
            W[k] = W[k] - self.learningRate*u_W[k]
            B[k] = B[k] - self.learningRate*u_B[k]

        # next batch
        i += self.batchSize
      acuu, loss = self.FindAccuracyAndLoss(W, B, self.X_train, self.Y_train)
      v_acc, v_loss = self.FindAccuracyAndLoss(W, B, self.X_val, self.Y_val)
      if self.isWandb == True:
        wandb.log({'accuracy': acuu})
        wandb.log({'loss': loss})
        wandb.log({'v_accuracy': v_acc})
        wandb.log({'v_loss': v_loss})
        wandb.log({'epoch': iteration})
      print(acuu, loss, v_acc, v_loss)
      iteration += 1

    self.Weights = W
    self.Baises  = B

  def NestrovBasedGradientDescent(self):
    iteration = 0
    W, B, preActivation, activation = self.Initialise()
    u_W, u_B = self.InitialiseEmptyWeightsAndBiases()
    # initializing u to be zero

    while(iteration < self.epochs):
      i = 0
      while i < self.X_train.shape[1]:

        y = self.FeedForward(self.X_train[:, i:i+self.batchSize], W, B, preActivation, activation)
        dw, db = self.BackWardPropogation(self.X_train[:, i:i+self.batchSize], self.Y_train[i:i+self.batchSize], W, preActivation, activation, y)

        for k in range(1, self.layers+1):
            u_W[k] = u_W[k]*self.betha + dw[k]
            u_B[k] = u_B[k]*self.betha + db[k]

        for k in range(1, self.layers+1):
            W[k] = W[k] - self.learningRate*(self.betha* u_W[k]+ dw[k])
            B[k] = B[k] - self.learningRate*(self.betha* u_B[k]+ db[k])

        i += self.batchSize
      acuu, loss = self.FindAccuracyAndLoss(W, B, self.X_train, self.Y_train)
      v_acc, v_loss = self.FindAccuracyAndLoss(W, B, self.X_val, self.Y_val)
      if self.isWandb == True:
        wandb.log({'accuracy': acuu})
        wandb.log({'loss': loss})
        wandb.log({'v_accuracy': v_acc})
        wandb.log({'v_loss': v_loss})
        wandb.log({'epoch': iteration})
      print(acuu, loss, v_acc, v_loss)
      iteration += 1

    self.Weights = W
    self.Baises  = B

  def RMSPROP(self):
    iteration = 0
    epochs = self.epochs
    layers = self.layers
    batchSize = self.batchSize
    betha = self.betha
    W, B, preActivation, activation  = self.Initialise()
    v_W, v_B = self.InitialiseEmptyWeightsAndBiases()
    # inititialising u to be zero

    while(iteration < epochs):
      i = 0
      while i < self.X_train.shape[1]:
        y = self.FeedForward(self.X_train[:, i:i+batchSize], W, B, preActivation, activation)
        dw, db = self.BackWardPropogation(self.X_train[:, i:i+batchSize], self.Y_train[i:i+batchSize], W, preActivation, activation, y)

        # update the v values with the gradient
        for k in range(1, layers+1):
          v_W[k] = v_W[k]*betha + (1 - betha) * (dw[k] ** 2)
          v_B[k] = v_B[k]*betha + (1 - betha) * (db[k] ** 2)

        # we will update the weights now with the momentum
        for k in range(1, layers+1):
          W[k] = W[k] - (self.learningRate/np.sqrt(v_W[k] + self.epsilon))*dw[k]
          B[k] = B[k] - (self.learningRate/np.sqrt(v_B[k] + self.epsilon))*db[k]

        i += batchSize
      acuu, loss = self.FindAccuracyAndLoss(W, B, self.X_train, self.Y_train)
      v_acc, v_loss = self.FindAccuracyAndLoss(W, B, self.X_val, self.Y_val)
      if self.isWandb == True:
        wandb.log({'accuracy': acuu})
        wandb.log({'loss': loss})
        wandb.log({'v_accuracy': v_acc})
        wandb.log({'v_loss': v_loss})
        wandb.log({'epoch': iteration})
      print(acuu, loss, v_acc, v_loss)
      iteration += 1

    self.Weights = W
    self.Baises  = B

  def ADAM(self):
    iteration = 0
    epochs = self.epochs
    layers = self.layers
    batchSize = self.batchSize

    W, B, preActivation, activation  = self.Initialise()
    v_W, v_B = self.InitialiseEmptyWeightsAndBiases()
    m_W, m_B = self.InitialiseEmptyWeightsAndBiases()
    mhat_W, mhat_B = self.InitialiseEmptyWeightsAndBiases()
    vhat_W, vhat_B = self.InitialiseEmptyWeightsAndBiases()
    # inititialising u to be zero
    t = 1

    while(iteration < epochs):
      # this is used to compute the gradients
      i = 0
      while i < self.X_train.shape[1]:
        y = self.FeedForward(self.X_train[:, i:i+batchSize], W, B, preActivation, activation)
        dw, db = self.BackWardPropogation(self.X_train[:, i:i+batchSize], self.Y_train[i:i+batchSize], W, preActivation, activation, y)

        # updating the momentum
        for k in range(1, layers+1):
          m_W[k] = self.betha1*m_W[k] + (1 - self.betha1)*dw[k]
          m_B[k] = self.betha1*m_B[k] + (1 - self.betha1)*db[k]

          # finding m hat of W and B
          mhat_W[k] = m_W[k]/(1 - self.betha1 ** t)
          mhat_B[k] = m_B[k]/(1 - self.betha1 ** t)

        # update the v values with the gradient
        for k in range(1, layers+1):
          v_W[k] = v_W[k]*self.betha2 + (1 - self.betha2) * (dw[k] ** 2)
          v_B[k] = v_B[k]*self.betha2 + (1 - self.betha2) * (db[k] ** 2)

          # finding v hat of W and B
          vhat_W[k] = v_W[k]/(1 - self.betha2 ** t)
          vhat_B[k] = v_B[k]/(1 - self.betha2 ** t)

        # we will update the weights now with the momentum
        for k in range(1, layers+1):
          l2_norm_w = np.linalg.norm(vhat_W[k])
          l2_norm_b = np.linalg.norm(vhat_B[k])
          W[k] = W[k] - (self.learningRate/np.sqrt(l2_norm_w) + self.epsilon)*mhat_W[k]
          B[k] = B[k] - (self.learningRate/np.sqrt(l2_norm_b) + self.epsilon)*mhat_B[k]

        t += 1
        i += self.batchSize

      acuu, loss = self.FindAccuracyAndLoss(W, B, self.X_train, self.Y_train)
      v_acc, v_loss = self.FindAccuracyAndLoss(W, B, self.X_val, self.Y_val)
      if self.isWandb == True:
        wandb.log({'accuracy': acuu})
        wandb.log({'loss': loss})
        wandb.log({'v_accuracy': v_acc})
        wandb.log({'v_loss': v_loss})
        wandb.log({'epoch': iteration})
      print(acuu, loss, v_acc, v_loss)
      iteration += 1

    self.Weights = W
    self.Baises  = B

  def NADAM(self):
    iteration = 0
    epochs = self.epochs
    layers = self.layers
    W, B, preActivation, activation  = self.Initialise()
    v_W, v_B = self.InitialiseEmptyWeightsAndBiases()
    m_W, m_B = self.InitialiseEmptyWeightsAndBiases()
    mhat_W, mhat_B = self.InitialiseEmptyWeightsAndBiases()
    vhat_W, vhat_B = self.InitialiseEmptyWeightsAndBiases()
    # inititialising u to be zero
    t = 1

    while(iteration < self.epochs):
      # this is used to compute the gradients
      i = 0
      while i < self.X_train.shape[1]:
        y = self.FeedForward(self.X_train[:, i:i+self.batchSize], W, B, preActivation, activation)
        dw, db = self.BackWardPropogation(self.X_train[:, i:i+self.batchSize], self.Y_train[i:i+self.batchSize], W, preActivation, activation, y)

        # updating the momentum
        for k in range(1, layers+1):
          m_W[k] = self.betha1*m_W[k] + (1 - self.betha1)*dw[k]
          m_B[k] = self.betha1*m_B[k] + (1 - self.betha1)*db[k]

          # finding m hat of W and B
          mhat_W[k] = m_W[k]/(1 - self.betha1 ** t)
          mhat_B[k] = m_B[k]/(1 - self.betha1 ** t)

        # update the v values with the gradient
        for k in range(1, layers+1):
          v_W[k] = v_W[k]*self.betha2 + (1 - self.betha2) * (dw[k] ** 2)
          v_B[k] = v_B[k]*self.betha2 + (1 - self.betha2) * (db[k] ** 2)

          # finding v hat of W and B
          vhat_W[k] = v_W[k]/(1 - self.betha2 ** t)
          vhat_B[k] = v_B[k]/(1 - self.betha2 ** t)

        # we will update the weights now with the momentum
        for k in range(1, layers+1):
          l2_norm_w = np.linalg.norm(vhat_W[k])
          l2_norm_b = np.linalg.norm(vhat_B[k])
          W[k] = W[k] - (self.learningRate/np.sqrt(l2_norm_w) + self.epsilon)*(mhat_W[k]*self.betha1 + (1 - self.betha1)*dw[k]/(1 - self.betha1 ** t))
          B[k] = B[k] - (self.learningRate/np.sqrt(l2_norm_b) + self.epsilon)*(mhat_B[k]*self.betha1 + (1 - self.betha1)*db[k]/(1 - self.betha1 ** t))

        t += 1
        i += self.batchSize

      acuu, loss = self.FindAccuracyAndLoss(W, B, self.X_train, self.Y_train)
      v_acc, v_loss = self.FindAccuracyAndLoss(W, B, self.X_val, self.Y_val)
      if self.isWandb == True:
        wandb.log({'accuracy': acuu})
        wandb.log({'loss': loss})
        wandb.log({'v_accuracy': v_acc})
        wandb.log({'v_loss': v_loss})
        wandb.log({'epoch': iteration})
      print(acuu, loss, v_acc, v_loss)
      iteration += 1

    self.Weights = W
    self.Baises  = B

  def fit(self):
    if self.optimiser == "sgd":
      self.SGD()
    if self.optimiser == "momentum":
      self.MomentBasedGradientDecent()
    if self.optimiser == "nag":
      self.NestrovBasedGradientDescent()
    if self.optimiser == "rmsprop":
      self.RMSPROP()
    if self.optimiser == "adam":
      self.ADAM()
    if self.optimiser == "nadam":
      self.NADAM()

  def confusionMatrix(self):
    # on the test data set
    predictions = self.predict(self.X_test)
    if self.dataset == "fashion_mnist":
      class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    else:
      class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    if self.isWandb == True:
      conf_matrix = confusion_matrix(self.Y_test, predictions)
      plt.figure(figsize=(10, 7))
      sns_heatmap = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                xticklabels=class_names, yticklabels=class_names)
      plt.title('Confusion Matrix')
      plt.ylabel('True Label')
      plt.xlabel('Predicted Label')

      # Save the plot to an image file
      heatmap_image_filename = "confusion_matrix_heatmap.png"
      plt.savefig(heatmap_image_filename)
      plt.close()  # Close the plot to avoid displaying it in the notebook/output

      # Log the image to Wandb
      wandb.log({"confusion_matrix_custom": wandb.Image(heatmap_image_filename)})

    else:
      conf_matrix = confusion_matrix(self.Y_test, predictions)
      plt.figure(figsize=(10, 7))
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                  xticklabels= class_names,
                  yticklabels= class_names)
      plt.title('Confusion Matrix')
      plt.ylabel('True Label')
      plt.xlabel('Predicted Label')
      plt.show()

# Neural network class end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs23m065")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Assignment 1")
    parser.add_argument("--dataset", "-d", help = "dataset", choices=["mnist","fashion_mnist"], default="fashion_mnist")
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=16)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "sgd", choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
    parser.add_argument("--loss","-l", default= "cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("--learning_rate","-lr", default=0.1, type=float)
    parser.add_argument("--momentum","-m", default=0.5,type=float)
    parser.add_argument("--beta","-beta", default=0.5, type=float)
    parser.add_argument("--beta1","-beta1", default=0.5,type=float)
    parser.add_argument("--beta2","-beta2", default=0.5,type=float)
    parser.add_argument("--epsilon","-eps",type=float, default = 0.000001)
    parser.add_argument("--weight_decay","-w_d", default=0.0,type=float)
    parser.add_argument("-w","--weight_init", default="random",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int, default=1)
    parser.add_argument("--hidden_size","-sz",type=int, default=4)
    parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","relu"], default="sigmoid")
    #parser.add_argument()
    
    
    
    args = parser.parse_args()
    # print(args.dataset)
    # print(args.epochs)
    # print(args.batch_size)
    # print(args.optimizer)
    # print(args.loss)
    # print(args.learning_rate)
    # print(args.momentum)
    # print(args.beta)
    # print(args.beta1)
    # print(args.beta2)
    # print(args.epsilon)
    # print(args.weight_decay)
    # print(args.weight_init)
    # print(args.num_layers)
    # print(args.hidden_size)
    # print(args.activation)

    wandb.login()
    wandb.init(project=args.wandb_project,entity=args.wandb_project)
    model = NeuralNetwork(inputSize = 784, hiddenLayers = args.num_layers, 
                          outputSize = 10, sizeOfHiddenLayers = args.hidden_size, 
                          batchSize = args.batch_size, learningRate = args.learning_rate, 
                          initialisationType = args.weight_init, optimiser = args.optimizer, 
                          activationFunc=args.activation,weightDecay = args.weight_decay,
                          isWandb = True, lossFunc = args.loss, epochs = 10, dataset = args.dataset)
    model.fit()
    wandb.finish()

