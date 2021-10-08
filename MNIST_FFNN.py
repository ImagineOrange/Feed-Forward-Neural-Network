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
            ax[row,col].set_title(f"Label: {y[counter]} - Predicted: {predictions[counter]}",fontsize=10)
            counter+=1      
    plt.suptitle(f"Sample Classifications from Final Training Minibatch --- Final Accuracy: {100*round(accuracy,2)}%")

def progress_bar(i,ceiling,loss,accuracy,key):
      barlength=25
      percentage = int(i/ceiling * 100) + 1
      num_hash = int(percentage/100 * barlength)
      num_space = int(barlength - num_hash)
      tally = '#' * num_hash
      space = '_' * num_space
    
      if key == 'backprop':
        sys.stdout.write(
            f"\rTraining Progress: [{tally}{space}] {round(percentage,5)}% - ({i}/{ceiling}) - " 
            f"accuracy: {round(accuracy * 100,2)} %       "
            )

def normalization(X): #normalize X to 0-1 --- Is this the best way to do this? 
     return  X / 255
     

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


def naive_backprop(epochs=5000,minibatch=4096,learning_rate=0.005): #at this point, we are not splitting data into train/test
    #Draw network, give stats
    print("\n\nThis FFNN is chugging along to classify handwritten digits of the MNIST dataset. \
           \ndataset source: http://yann.lecun.com/exdb/mnist/ \
           \n\nFeed-forward Network Shape: 32 x 16 x 10 neurons               25,818 Tunable Parameters \n")
    print("Input Vector    ···***"+32*'x'+"***···   784 features (pixels per img)")
    print("Hidden 1              "+32*'*'+"         784 inputs, 32 outputs")
    print("Hidden 2              "+8*' '+16*'*'+12*" "+"      32 inputs, 16 outputs"+"")
    print("Output                "+11*' '+10*'*'+15*' '+"      16 inputs, 10 outputs\n\
        \nepochs =", epochs, ", minibatch size =", minibatch, ", learning_rate =",learning_rate,"\n")
        
    #Fetch training data --- reshape to row vectors for each digit
    X = fetch_MNIST("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1,784)) 
    y = fetch_MNIST("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]      
    
    #Normalize X for faster computations
    #X = normalization(X)

    #Shuffle Dataset
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]

    #init model layers
    input_layer = Layer_Dense(784,32) #input layer
    hidden_layer = Layer_Dense(32,16) #hidden layer
    output_layer = Layer_Dense(16,10) #output layer
    
    #init model activations
    input_activation = Activation_ReLU() 
    hidden_activation = Activation_ReLU()
    output_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
   
    #init model optimizer function
    optimizer = Stochastic_Gradient_Descent(learning_rate) #SGD - learning rate

    #for plotting
    loss_ = []
    accuracy_ = []
    epochs_ = []
    
    for epoch in range(epochs): #training loop for SGD 
        
        #select random rows of digit matrix
        number_of_rows = X.shape[0]
        random_indices = np.random.choice(number_of_rows, size=minibatch, replace=False)

        #compute mini-batches for Gradient Descent
        X = X[random_indices,:] 
        y = y[random_indices]

        #forward pass
        input_layer.forward_pass(X)  #forward pass input layer
        input_activation.forward_pass(input_layer.output) #forward pass input activation
        hidden_layer.forward_pass(input_activation.output) #forward pass thru hidden
        hidden_activation.forward_pass(hidden_layer.output) #forward pass thru hidden activation
        output_layer.forward_pass(hidden_activation.output) #forward pass thru output activation
        loss = output_activation.forward_pass(output_layer.output,y) #forward pass thru output layer
        
        #Calculate accuracy from output layer activations and targets
        predictions = np.argmax(output_activation.output, axis=1)
        if len(y.shape)==2:
            y = np.argmax(y,axis=1)
        accuracy = np.mean(predictions==y)

        #Backward pass
        output_activation.backward_pass(output_activation.output,y) #gradient of loss/softmax
        output_layer.backward_pass(output_activation.dinputs) #gradient of output layer activation
        hidden_activation.backward_pass(output_layer.dinputs) #gradient of hidden activation
        hidden_layer.backward_pass(hidden_activation.dinputs) #gradient of hidden layer
        input_activation.backward_pass(hidden_layer.dinputs) #gradient of input activation function
        input_layer.backward_pass(input_activation.dinputs) #gradient of input activations

        optimizer.update_parameters(input_layer) #update layer parameters using gradients calculated during backprop
        optimizer.update_parameters(hidden_layer)
        optimizer.update_parameters(output_layer)
        
        #progress bar
        progress_bar(epoch,epochs,loss,accuracy,key='backprop') #progress bar :)
        sys.stdout.flush()
        
        #for plotting
        loss_.append(loss)
        accuracy_.append(accuracy)
        epochs_.append(epoch)


    #plot input layer weights
    plt.figure(figsize = (12,7))
    plt.imshow(input_layer.weights, aspect='auto', interpolation='none',cmap='jet')
    plt.title("Strength of Edges between Input Features and First Hidden Layer")
    plt.xlabel("Neurons of First Hidden Layer")
    plt.ylabel("Input features of Input Vector")
    plt.colorbar()

    #plot hidden layer weights
    plt.figure(figsize = (12,7))
    plt.imshow(hidden_layer.weights, aspect='auto', interpolation='none',cmap='jet')           
    plt.title("Strength of Edges between First and Second Hidden Layers")
    plt.xlabel("Neurons of Second Hidden Layer")
    plt.ylabel("Neurons of First Hidden Layer")
    plt.colorbar()
    
    #plot output layer weights
    plt.figure(figsize = (12,7))
    plt.imshow(output_layer.weights, aspect='auto', interpolation='none',cmap='jet')
    plt.title("Strength of Edges between Second Hidden Layer and Output Layer")
    plt.xlabel("Neurons of Output Layer")
    plt.ylabel("Neurons of Second Hidden Layer")
    plt.colorbar()

    #normalize and plot loss / accuracy over time
    np.arange(0,epochs)
    loss_ = (loss_ - min(loss_))/(max(loss_) - min(loss_)) #normalize
    accuracy_ = (accuracy_ - min(accuracy_))/(max(accuracy_) - min(accuracy_)) 
    plt.figure(figsize = (12,7))
    plt.title("Loss and Accuracy over Epochs of Gradient Descent")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Cost / Accuracy")

    plt.scatter(epochs_,loss_,c='red',s=1,label="Network Loss")
    plt.scatter(epochs_,accuracy_,c='blue',s=1,label="Classification Accuracy")
    plt.legend()
    
    #Visualize Example digits
    visualize_digits(X,y,predictions,accuracy)
    
    # #Visualize correct vs. incorrect classifications!
    # plt.figure(figsize=(12,7))
    # plt.title("Classification Matrix")
    # plt.imshow(np.array(predictions==y).reshape(64,64),cmap='jet')
    # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

    print("\n\nDone! Final training accuracy:",100*round(accuracy,2), "% --- Best training accuracy: 99.02%\n")


#main - coordinate model
def main():

    #Model with backprop
    naive_backprop() #epochs=10000,minibatch=5000,learning_rate=0.005
    end = time.time()
    print(f"Full Program runtime: {round((end-start),3)} s\n\n")

main()
plt.show()


