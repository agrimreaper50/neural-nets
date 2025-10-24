import numpy as np

np.random.seed(42)
N = 100
X = np.random.uniform(-5, 5, (N, 1))

Y = 3 * X + 1

def linear(z):
    return z

def mse_loss(y_hat, y):
    diff = y_hat - y 
    return np.mean(diff ** 2)

D_in = 1
H = 5
D_out = 1

rng = np.random.RandomState(42)
W1 = rng.randn(D_in, H) * 0.5

b1 = np.zeros((1, H))
W2 = rng.randn(H, D_out) * 0.5
b2 = np.zeros((1, D_out))

lr = 0.01
epochs = 2000


def forward(X, W1, b1, W2, b2):
    z1 = X.dot(W1) + b1
    h = np.tanh(z1)
    z2 = h.dot(W2) + b2
    y_hat = linear(z2)
    return y_hat, h

for epoch in range(1, epochs + 1):
    y_hat, h = forward(X, W1, b1, W2, b2)
    loss = mse_loss(y_hat, Y)

    delta2 = (y_hat - Y) / N
    dW2 = h.T.dot(delta2)
    db2 = delta2.sum(axis=0, keepdims=True)

    delta1 = delta2.dot(W2.T) * (1 - h**2)
    dW1 = X.T.dot(delta1)
    db1 = delta1.sum(axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
        print("Y_hat: " + str(y_hat))

