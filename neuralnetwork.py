# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:51:52 2017

@author: jkligon
"""
import numpy as np
import scipy.special

#neural network class definition
class neuralNetwork:
    
    #initialise the neural network
    #set the number of input, hidden, and output nodes
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        #set the number of nodes for input, hidden, and output nodes
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        
        #set the learning rate
        self.lr = learningRate
        
        #link weight matrices, weight_input_hidden (wih) and weight_hidden_output (who)
        #weights inside the arrays w_i_j, where link is from node i to j in the next layer
        """
        [(w11 w21
          w12 w22)]
        """
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        
        #activation function using the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    #refine the weights after being given a training set example to learn from
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin = 2) .T
        targets = np.array(targets_list, ndmin = 2) .T
        
        #calculate siginals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        
        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        #error is (target - actual):This for weights between the hidden and final layer
        output_errors = targets - final_outputs
        
        #hidden layer error is the output_errors, split, by weights, recombined at hidden nodes
        #weights between the input and hidden layers
        hidden_errors = np.dot(self.who.T, output_errors)
        
        #update the weights for links between the hidden and output layers
        self.who = self.who + self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        #update the weights for links between the input and hidden layers
        self.wih   = self.wih + self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    
    #give an answer from the output nodes after being given an input
    def query(self, inputs_list):
        #convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin = 2) .T
        
        #calculate ssignals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        
        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate singlas into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
