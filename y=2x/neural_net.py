import numpy as np

np.random.seed(42)
N = 100
x = np.random.uniform(-5, 5, N) # inputs sampled from uniform (-5, 5)

y = 2.0 * x # the true output, (no noise for simplicity)

'''
the seeding fixes random numbers so results are reproducible

N sets how many sample points we create
'''

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


def forward(x, w, b):
    return x.dot(w) + b


def mse_loss(y_hat, y):
    diff = y_hat - y
    return np.mean(diff ** 2)


def compute_gradients(x, y, y_hat):
    '''
    x: (N, 1), y: (N, 1), y_hat: (N, 1)
    Returns:
    dw -> shape (1, 1)
    db -> shape (1, )
    '''
    N = x.shape[0]

    error = y_hat - y 
    dw = (2.0 / N) * (x * error).sum(axis=0, keepdims=True).T
    db = (2.0 / N) * error.sum(axis=0)                
    return dw, db

lr = 0.01
epochs = 200
val = 20

w = np.random.randn(1, 1) * 0.1
b = np.zeros((1, ))

for epoch in range(1, epochs + 1):
    y_hat = forward(x, w, b)

    loss = mse_loss(y_hat, y)

    dw, db = compute_gradients(x, y, y_hat)

    w -= lr * dw
    b -= lr * db
    
    if epoch % val == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {loss:.6f} | w: {w.ravel()[0]:.4f} | b: {b.ravel()[0]:.4f}")


print("\nTraining finished.")
print(f"Learned w = {w.ravel()[0]:.6f}, b = {b.ravel()[0]:.6f}")
test_x = np.array([[-10.0],[0.0],[5.5]])
preds = forward(test_x, w, b)
for xv, pv in zip(test_x.ravel(), preds.ravel()):
    print(f"x = {xv:6.2f} | prediction = {pv:8.4f} | true = {2*xv:8.4f}")

