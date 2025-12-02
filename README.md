A project from 2021

# Feed-Forward-Neural-Network
Feed-Forward Neural Network from scratch using numpy. Classifies digits of the MNIST dataset
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

This is a feed-forward neural network written in python using only numpy and some accessory packages for visualization. 
Run the program from a full-screen console window, and follow the example parameters upon first run to get a feel for it!

The dataset used for training/testing can be found here http://yann.lecun.com/exdb/mnist/

The training algorithm is Stochastic Gradient Descent with momentum, and learning rate decay. Maybe in the future I will implement some sort of regularization.
Each iteration of training consists of one forward pass (prediction), a validation test which consists of a prediction of 10,000 unseen digits, 
and then a backward pass in which the gradient for the training examples is calculated and its constituent updates applied to neuron weights/biases

I used the book 'Neural Networks from Scratch in python' by Harrison Kingsley and Daniel Kukiela as a reference!



The default 'example' parameters result in a 98.02% accuracy on the validation set on my machine.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Added 4/25:
Revisited after a few years and used code to classify a spiral dataset for the purpose of drawing decision boundaries.

Very cool looking!

![decision_boundary](https://github.com/user-attachments/assets/80a2b0db-133d-4cad-8e28-4b24b4ea12fc)
