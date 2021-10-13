#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:52:52 2021

@author: ecrouse
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time 

start = time.time()

#This MNIST fetching function was written by George Hotz:
#https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
def fetch_MNIST(url):
  import requests, gzip, os, hashlib
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  
  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def visualize_digits(X,y,predictions,accuracy): #visualize some digits
    fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(12,7))
    #back to square
    X = X.reshape(X.shape[0],28,28)
    counter = 0
    for row in range(ax.shape[0]):
        for col in range(ax.shape[1]):
            ax[row,col].imshow(X[counter],aspect='equal')
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
            ax[row,col].set_title(f"Label: {y[counter]} - Predicted: {predictions[counter]}",fontsize=9)
            counter+=1      
    plt.suptitle(f"Sample Classifications from Test Dataset --- Final Test Accuracy: {100*round(accuracy,2)}%")

def progress_bar(i,ceiling,train_accuracy,test_accuracy):
      barlength=25
      percentage = int(i/ceiling * 100) + 1
      num_hash = int(percentage/100 * barlength)
      num_space = int(barlength - num_hash)
      tally = '#' * num_hash
      space = '_' * num_space
    
      sys.stdout.write(
        f"\rTraining Progress: [{tally}{space}] {round(percentage,5)}% - ({i}/{ceiling}) - " 
        f"batch acc: {round(train_accuracy * 100,4)} % --- "
        f"val acc: {round(test_accuracy * 100,4)} %  ")

def draw_output(epochs,minibatch,learning_rate): #Draw network, give stats
    print("\n\nThis FFNN is chugging along to classify handwritten digits of the MNIST dataset. \
           \ndataset source: http://yann.lecun.com/exdb/mnist/ \
           \n\nFeed-forward Network Shape: 784 x 128 x 64 x 10 neurons: \n")
    print("Input        ···"+26*'x'+"···         784 features (pixels per img)")
    print("Hidden 1     "+6*' '+20*'*'+12*" "+"   784 inputs, 128 neurons"+"")
    print("Hidden 2               "+12*'*'+"                   128 inputs, 64 neurons")
    print("Output       "+13*' '+6*'*'+15*' '+"        64 inputs, 10 neurons\n\
        \nepochs =", epochs, ", batch size =",minibatch, ", learning_rate =",learning_rate,"\n")
    print("Training Algorithm: Batch Gradient Descent")
        
    #Fetch training data --- reshape to row vectors for each digit
    X = fetch_MNIST("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1,784)) 
    y = fetch_MNIST("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]     


#This is a really cool network drawing function written by Colin Raffel 
#https://gist.github.com/craffel/2d727968c3aaebd10359
def draw_neural_net(left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle("  128 x 64 x 10")
    ax = fig.gca()
    ax.axis('off')
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=3)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k',linewidth = .04)
                ax.add_artist(line)

def normalize(X):
    return X / 255

#Fullly connected layer
class Layer_Dense: 
    #initialize layer
    def __init__(self,n_inputs,n_neurons): #n inputs by n neurons
        self.weights = 0.1*np.random.randn(n_inputs,n_neurons) #initialize weights randomly
        self.biases = np.zeros((1,n_neurons))
    
    #Forward pass
    def forward_pass(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        self.inputs = inputs #for derivative calculations
    
    #Backward pass
    def backward_pass(self,dvalues):
        #Calculate the Gradient of parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues,axis = 0,keepdims = True)
        #Calcualte the Gradient on input values
        self.dinputs = np.dot(dvalues,self.weights.T)

#rectified linear unit activation function (threshold function: binary output)
class Activation_ReLU: 
    #Forward pass
    def forward_pass(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs) #if x>0, return x, else return 0
    
    #Backward pass
    def backward_pass(self,dvalues):
        #modify original variable, first copy list
        self.dinputs = dvalues.copy()
        #0s where values were negative or 0:
        self.dinputs[self.inputs <= 0] = 0
    
#outputs a probability distribution - exponentiated and normalized output
class Activation_Softmax: 
    #Forward pass
    def forward_pass(self,inputs):
        #we need to find max of batch, not total!
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) 
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities
    def backward_pass(self,dvalues):
        #create empty array
        self.dinputs = np.empty_like(dvalues)
        #enumerate outputs and gradients
        for index, (single_output,single_dvalues) in \
            enumerate(zip(self.output,dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1,1)
            #Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output,single_output.T)
            #Calculate sample-wise Gradient
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

#Calculate loss
class Loss: 
    #Parent loss function, daughter functions will inherit from this class
    def calculate(self,output,y):
        sample_losses = self.forward_pass(output,y)
        batch_loss = np.mean(sample_losses)
        return batch_loss
        
#Categorical Cross Entropy
class Loss_Categorical_Cross_Entropy(Loss): #inherits from Loss class
    #Forward pass
    def forward_pass(self,y_pred,y_true):
        #number of samples in a batch
        samples = len(y_pred)
        #clip data to prevent division by zero / will prevent mean from being dragged to an outlier
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7) #high/low values clipped to edge
        #probabilities for target values
        
        if len(y_true.shape) == 1: #if not one-hot encoded
            #grab target elements via scalar class values
            correct_confidences = y_pred_clipped[range(samples),y_true]
        
        #mask values if one-hot encoded
        elif len(y_true.shape) == 2:
            #grab target elements via one-hot encoding matrix
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        #calculate loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    #Backward pass
    def backward_pass(self,dvalues,y_true):
        ## of samples
        samples = len(dvalues)
        ## of values in each sample
        labels = len(dvalues[0])
        #if labels are sparse, turn into one-hot vector
        if len(y_true.shape)==1:
            y_true = np.eye(labels)[y_true]
        
        #Gradient calculation:
        self.dinputs = -y_true / dvalues 
        #normalize Gradient:
        self.dinputs = self.dinputs / samples

#Categorical cross entropy with Softmax - there's a formula to calculate the gradient for both steps simultaneously
class Activation_Softmax_Loss_CategoricalCrossentropy():
    #Initialize Softmax and CCE objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Cross_Entropy()
    #Forward pass
    def forward_pass(self,inputs,y_true):
        #output layer's softmax activation function 
        self.activation.forward_pass(inputs)
        #init output
        self.output = self.activation.output
        #Calculate Loss
        return self.loss.calculate(self.output,y_true)
    #Backward pass
    def backward_pass(self,dvalues,y_true):
        ## samples
        samples = len(dvalues)
        #if labels are one-hot encoded turn them into discrete values 
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
        #Copy for mod
        self.dinputs = dvalues.copy()
        #Calc gradient
        self.dinputs[range(samples),y_true] -= 1
        #normalize gradient
        self.dinputs = self.dinputs / samples

#Stochastic Gradient Descent
class Stochastic_Gradient_Descent:
    #Initialize the optimizer object and set default learning rate - (step size for SGD)
    def __init__(self,Learning_rate):
        self.learning_rate = Learning_rate
    #Update parameters 
    def update_parameters(self,layer):
        layer.weights += -self.learning_rate * layer.dweights #multiply learning rate by final layer gradients
        layer.biases += -self.learning_rate * layer.dbiases 

    #################### - FFNN model and Training below - ####################

def minibatch_GD(X_train,y_train,X_test,y_test,epochs,minibatch,learning_rate): #at this point, we are not splitting data into train/test
    #draw console output
    draw_output(epochs,minibatch,learning_rate)

    #Shuffle Dataset
    randomize = np.arange(len(X_train))
    np.random.shuffle(randomize)
    X_train = X_train[randomize]
    y_train = y_train[randomize]
    #X_train = normalize(X_train)

    randomize = np.arange(len(X_test))
    np.random.shuffle(randomize)
    X_test = X_test[randomize]
    y_test = y_test[randomize]
    #X_test = normalize(X_test)

    #init model layers
    input_layer = Layer_Dense(784,128) #input layer (first hidden)
    hidden_layer = Layer_Dense(128,64) #second hidden layer
    output_layer = Layer_Dense(64,10) #output layer
    
    #init model activations
    input_activation = Activation_ReLU() 
    hidden_activation = Activation_ReLU()
    output_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
   
    #init model optimizer function
    optimizer = Stochastic_Gradient_Descent(learning_rate) #SGD - learning rate
    
    #training----------------------------------------------------------------------------------

    training_accuracy = []
    testing_accuracy = []

    for epoch in range(epochs): #training loop for SGD 
        
        #select random rows of digit matrix
        number_of_rows = X_train.shape[0]
        random_indices = np.random.choice(number_of_rows, size=minibatch, replace=False)

        #compute mini-batches for Gradient Descent
        X_train = X_train[random_indices,:] 
        y_train = y_train[random_indices]

        #forward pass
        input_layer.forward_pass(X_train)  #forward pass input layer
        input_activation.forward_pass(input_layer.output) #forward pass input activation
        hidden_layer.forward_pass(input_activation.output) #forward pass thru hidden
        hidden_activation.forward_pass(hidden_layer.output) #forward pass thru hidden activation
        output_layer.forward_pass(hidden_activation.output) #forward pass thru output activation
        
        #Calculate Loss
        loss = output_activation.forward_pass(output_layer.output,y_train) #forward pass thru output layer
        
        #Backward pass
        output_activation.backward_pass(output_activation.output,y_train) #gradient of loss/softmax
        output_layer.backward_pass(output_activation.dinputs) #gradient of output layer activation
        hidden_activation.backward_pass(output_layer.dinputs) #gradient of hidden activation
        hidden_layer.backward_pass(hidden_activation.dinputs) #gradient of hidden layer
        input_activation.backward_pass(hidden_layer.dinputs) #gradient of input activation function
        input_layer.backward_pass(input_activation.dinputs) #gradient of input activations

        #update layer parameters using gradients calculated during backprop
        optimizer.update_parameters(input_layer) 
        optimizer.update_parameters(hidden_layer)
        optimizer.update_parameters(hidden_layer)
        optimizer.update_parameters(output_layer)
        
        #Calculate accuracy from output layer activations and targets
        training_predictions = np.argmax(output_activation.output, axis=1)
        if len(y_train.shape)==2:
            y_train = np.argmax(y_train,axis=1)
            train_accuracy = np.mean(training_predictions==y_train)
        train_accuracy = np.mean(training_predictions==y_train)

        #Validate the network --------------------------------------------------------------------

        #select random rows of digit matrix
        number_of_rows = X_test.shape[0]
        random_indices = np.random.choice(number_of_rows, size=minibatch, replace=False)

        #compute mini-batches for Gradient Descent
        X_test = X_test[random_indices,:] 
        y_test = y_test[random_indices]

        #Forward pass for validation
        input_layer.forward_pass(X_test)  #forward pass input layer
        input_activation.forward_pass(input_layer.output) #forward pass input activation
        hidden_layer.forward_pass(input_activation.output) #forward pass thru hidden
        hidden_activation.forward_pass(hidden_layer.output) #forward pass thru hidden activation
        output_layer.forward_pass(hidden_activation.output) #forward pass thru output activation
        output_activation.forward_pass(output_layer.output,y_test)
        
        #Calculate accuracy from output layer activations and targets
        test_predictions = np.argmax(output_activation.output, axis=1)
        if len(y_test.shape)==2:
            y_test = np.argmax(y_test,axis=1)
            test_accuracy = np.mean(test_predictions==y_test)
        test_accuracy = np.mean(test_predictions==y_test)
        
        #progress bar
        progress_bar(epoch,epochs,train_accuracy,test_accuracy)
        sys.stdout.flush()
    
        training_accuracy.append(train_accuracy)
        testing_accuracy.append(test_accuracy)
    
    #plotting -----------------------------------------------------------------------------------------
    
    #visualize network
    draw_neural_net(.15, .85, 0, 1, [64, 32, 5]) #each layer is cut in half for better visual
    
    #plot input layer
    plt.figure(figsize=(12,7))
    plt.subplot(1,3,1)
    plt.imshow(input_layer.weights,cmap='jet')
    plt.title(f"Input Layer Weights")
    
    #middle
    plt.subplot(1,3,2)
    plt.imshow(hidden_layer.weights,cmap='jet')
    plt.title("Middle Hidden Layer Weights")

    #Plot output layer weights
    plt.subplot(1,3,3)
    plt.imshow(output_layer.weights,cmap='jet')
    plt.title("Output Layer Weights")
    

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.suptitle("Layer Edge Weights")

    #plot some example digits from validation set
    visualize_digits(X_test,y_test,test_predictions,test_accuracy)

    #plot training and validation accuracy over epochs od SGD
    epochs = np.arange(epochs)
    plt.figure(figsize=(12,7))
    plt.scatter(epochs,training_accuracy,c='red',s=3,label="Training Accuracy")
    plt.scatter(epochs,testing_accuracy,c='blue',s=3,label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy over SGD")
    plt.legend()

    #plot digit
    plt.figure(figsize=(12,7))
    plt.imshow(X_test[-1].reshape(28,28))
    plt.title(f"Label: {y_test[-1]}")
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);


#main - coordinate model
def main():
    #Fetch training data --- reshape to row vectors for each digit
    X_train = fetch_MNIST("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1,784)) 
    y_train = fetch_MNIST("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]      
    X_test = fetch_MNIST("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))
    y_test = fetch_MNIST("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    
    #Model with backprop
    minibatch_GD(
        X_train,y_train,
        X_test,y_test,
        epochs=25000,
        minibatch=10000,
        learning_rate=0.00025) #20000 epochs, 10000 minibatch, learning = 0.005
    
    end = time.time()
    print(f"\n--Full Program runtime-- {round((end-start),3)} s\n\n")

main()

plt.show()
