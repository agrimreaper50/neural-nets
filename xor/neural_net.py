import numpy as np

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

# Activation functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def tanh_grad(a):
    return 1.0 - a**2

# Loss
def bce_loss(y_hat, y):
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1.0 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Initialize parameters
rng = np.random.RandomState(42)
D_in = 2
H = 4
D_out = 1

W1 = rng.randn(D_in, H) * 1.0
b1 = np.zeros((1, H))
W2 = rng.randn(H, D_out) * 1.0
b2 = np.zeros((1, D_out))

lr = 0.1
epochs = 1000
N = X.shape[0]

for epoch in range(1, epochs+1):
    # Forward
    z1 = X.dot(W1) + b1
    h = tanh(z1)
    z2 = h.dot(W2) + b2
    y_hat = sigmoid(z2)

    # Loss
    loss = bce_loss(y_hat, Y)

    # Backprop
    delta2 = (y_hat - Y) / N
    dW2 = h.T.dot(delta2)
    db2 = delta2.sum(axis=0, keepdims=True)

    delta1 = delta2.dot(W2.T) * tanh_grad(h)
    dW1 = X.T.dot(delta1)
    db1 = delta1.sum(axis=0, keepdims=True)

    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0 or epoch == 1:
        preds = (y_hat > 0.5).astype(int)
        acc = (preds == Y).mean()
        print(f"Epoch {epoch:5d} | loss: {loss:.6f} | acc: {acc*100:.1f}%")

# Final predictions
z1 = X.dot(W1) + b1
h = tanh(z1)
z2 = h.dot(W2) + b2
y_hat = sigmoid(z2)

print("\nFinal predictions (probabilities):")
for x_row, p, y_true in zip(X, y_hat.ravel(), Y.ravel()):
    print(f"x={x_row.tolist()} -> p={p:.4f}  predicted={int(p>0.5)} true={int(y_true)}")
