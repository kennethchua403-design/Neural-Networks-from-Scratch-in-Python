import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0,weight_regularizer_l2=0, 
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims= True)  #Revise
        self.dinputs = np.dot(dvalues, self.weights.T) #Must it be in this order?

        #Gradients on regulatization
        #L1 on weights
        if self.weight_regularizer_l1 > 0: #turn on regularization
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1  #Else = 1
            self.dweights += self.weight_regularizer_l1 * dL1   #Derivatives of L1 Weights Formula

        #L2 on weights
        if self.weight_regularizer_l2 > 0: 
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights #Derivatives of L2 Weights Formula

        #L1 on bias
        if self.bias_regularizer_l1 > 0: 
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1  #Else = 1
            self.dbiases += self.bias_regularizer_l1 * dL1   #Formula for derivatives of L1 Weights

        #L2 on bias
        if self.bias_regularizer_l2 > 0: 
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases #Derivatives of L2 Weights Formula
        

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


# We will use the Adam Optimizer

class Optimizer_Adam:
    #Initialize optimizer to set settings
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations)) #Learning rate decreases to find Global Minimum
    
    def update_params(self, layer):
        #If layer does not contain cache arrays, create them and fill them with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        #Update momentum
        
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights

        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Momentum Bias Correction
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache
        layer.weight_cache = self.beta_2 * layer.weight_cache +  (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache +  (1 - self.beta_2) * layer.dbiases**2

        #Cache Bias Correction

        weight_cache_corrected = layer.weight_cache /  (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        #Update Weights
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected /  (np.sqrt(bias_cache_corrected) + self.epsilon)

    #Call once after 1 epoch
    def post_update_params(self):
        self.iterations +=1

class Loss:

    def regularization_loss(self, layer): #Forward Pass
        #0 by default
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0: #L1 Weight Regularization
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0: #L2 Weight Regularization
                    regularization_loss += layer.weight_regularizer_l2 * np.sum((layer.weights)**2)

        if layer.bias_regularizer_l1 > 0: #L1 Bias Regularization
                    regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        
        if layer.bias_regularizer_l2 > 0: #L2 Bias Regularization
                    regularization_loss += layer.bias_regularizer_l2 * np.sum((layer.biases)**2)
        
        return regularization_loss

    
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

#Wraps Softmax activation and cross-entropy Loss
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, true_class):      #Chains the forward function for softmax output and Loss
        self.activation.forward(inputs)

        self.output = self.activation.output  #Probabilities

        return self.loss.calculate(self.output, true_class)

    def backward(self, dvalues, true_class):  #Shortcut to calculate Gradient
        samples = len(dvalues)

        if len(true_class.shape) == 2:
            true_class = np.argmax(true_class, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), true_class] -=1

        self.dinputs = self.dinputs / samples
    
X, y = spiral_data(samples=300, classes=3)

#Build the Network
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
                    bias_regularizer_l2=5e-4)

activation1 = Activation_Relu()

dense2 = Layer_Dense(64, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#Create Optimizer
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7) ###

for epoch in range(10001): ###

    #Data Flow 
    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    Data_loss = loss_activation.forward(dense2.output, y)   #Data loss = Loss from Prediction

    #Calculate regulariation penalty
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2) ###Revise "layer."

    Total_loss = Data_loss + regularization_loss   

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y) ###

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'Total loss: {Total_loss: .3f} (' +
            f'Data loss: {Data_loss:.3f}, ' +
            f'Regularization loss: {regularization_loss:.3f}),' +
            f'Learing Rate: {optimizer.current_learning_rate}')

    

    #Backward Pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    #Use Testing data set to validate the model
    # Ensure Testing Loss identical with Training Loss
X_test, Y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

Data_loss = loss_activation.forward(dense2.output, Y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(Y_test.shape) == 2:
    Y_test = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predictions == Y_test) ###Revise

print(f"validation, acc: {accuracy:.3f}, Validation_Data_loss: {Data_loss:.3f}")
