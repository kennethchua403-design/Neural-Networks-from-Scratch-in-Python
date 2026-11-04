import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues):
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims= True)  #Revise
        self.dinputs = np.dot(dvalues, self.weights.T) #Must it be in this order?

class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()  #To prevent changing of original values
        self.dinputs[self.inputs < 0] = 0  #Stop Gradients with Inputs < 0 from passing through

class Activation_Softmax:

    def forward(self, inputs):
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) #Create unintialized array

        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):     #single_output is the probabilities from 'forward'

            single_output = single_output.reshape(-1,1)

            #Calculate the Jacobian Matrix
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues) #converts gradient with respect to the softmax input

class Loss:

    def calculate(self, output, true_class ):

        sample_loss = self.forward(output,true_class)    #Calls forward function in class Loss_CategoricalCrossentropy(Loss)

        #calculate mean loss
        data_loss = np.mean(sample_loss)

        return data_loss

class Loss_CategoricalCrossentropy(Loss):

    def forward(self, output, true_class):  
        samples = len(output)

        #Clip data to prevent division by 0
        output_clipped = np.clip(output, 1e-7, 1-1e-7)

        if len(true_class.shape) == 1:
            correct_confidences = output_clipped[range(samples), true_class]

        elif len(true_class.shape) == 2:
            correct_confidences = np.sum(output_clipped * true_class, axis=1)

        loss = -np.log(correct_confidences)
        return loss

    def backward(self, dvalues, true_class):
        samples = len(dvalues)

        labels = len(dvalues[0]) # number of classes (columns) per row

        if len(true_class.shape) == 1:
            true_class = np.eye(labels)[true_class]  #Creates indentity matrix according to the true_class position

            self.dinputs = -true_class/ dvalues     #Calculate gradients of the correct class according to derivatives of -log(dvalues) -> -1/dvalues
            self.dinputs = self.dinputs / samples   #Averages the Gradients over the batch loss

#Combined Softmax activation and cross-entropy Loss
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, true_class):  #inputs = dvalues
        self.activation.forward(inputs)

        self.output = self.activation.output

        return self.loss.calculate(self.output, true_class)

    def backward(self, dvalues, true_class):
        samples = len(dvalues)

        if len(true_class.shape) == 2:
            true_class = np.argmax(true_class, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), true_class] -=1

        self.dinputs = self.dinputs / samples
    
X, y = spiral_data(samples=100, classes=3)

#Build the Network
dense1 = Layer_Dense(2, 3)

activation1 = Activation_Relu()

dense2 = Layer_Dense(3, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#Data Flow 
dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y)

print(loss_activation.output[:5])

#Loss Value
print("loss value:", loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions ==y)

#Accuracy
print("Accuracy:", accuracy)

#Backward Pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
