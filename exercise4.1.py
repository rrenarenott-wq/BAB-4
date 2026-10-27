# Example 4.1 Neuron.py - Modifikasi untuk Exercise 4.1 (Logika OR)
from numpy import array, exp, dot, random

# Set up parameters ==========================================
# Input: [0, 0], [1, 1], [1, 0], [0, 1]
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])

# Output untuk Logika OR: 0, 1, 1, 1
outputs = array([[0, 1, 1, 1]]).T

random.seed(1)
weights = 2 * random.random((2, 1)) - 1

# Define a Single Neuron function ==============================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

# Train the Neuron =============================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron ==============================================
# Menguji neuron dengan input [1, 0] yang seharusnya menghasilkan output mendekati 1
x = array([1, 0])
print(neuron(x, weights))