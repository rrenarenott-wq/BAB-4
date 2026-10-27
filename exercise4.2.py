# Example 4.2 Multiple Layer Perceptron - Modified for 3 inputs
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

# Modifikasi X menjadi 3 input (fitur) per sampel data latih
X = [[0., 0., 0.], [1., 1., 1.], [0., 1., 0.], [1., 0., 1.]]
y = [0, 1, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

# Modifikasi data uji (predict) agar memiliki 3 input juga
print(clf.predict([[2., 2., 2.], [-1., -2., -1.]]))
print([coef.shape for coef in clf.coefs_])