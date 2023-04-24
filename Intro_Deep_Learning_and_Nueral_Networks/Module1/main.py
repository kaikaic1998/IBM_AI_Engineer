# Initialize a network exercise

import numpy as np # import the Numpy library
from random import seed
import numpy as np

# Function for initialize networks
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network

# Function for computing weight sum of a node
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

# Function for computing activation function of a node
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

# Function for computing Forward Propagation
def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions

# ----------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------Construct a network------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------

# 1. takes 5 inputs
# 2. has three hidden layers
# 3. has 3 nodes in the first layer, 2 nodes in the second layer, and 3 nodes in the third layer
# 4. has 1 node in the output layer
small_network = initialize_network(5, 3,[3, 2, 3], 1)

# Generate 5 random inputs
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('The inputs to the network are {}'.format(inputs))

# Compute the prediction (output) of this small_network
predictions = forward_propagate(small_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))