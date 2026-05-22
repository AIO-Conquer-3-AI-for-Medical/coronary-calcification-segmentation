import matplotlib.pyplot as plt
import numpy as np

X = np.array([[-1], [-2], [1], [2], [3], [4], [5], [6], [7]])
y = np.array([[2.20], [2.78], [1.27], [12.70], [40.17], [89.92], [168.0], [278.36], [430.29]])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X, np.square(X), X ** 3), axis=1)

# loss function
def loss(w):
    e = Xbar @ w - y
    return 0.5 * np.sqrt(e ** 2)

def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi * a).reshape(4, 1)

def SGD(w_init, grad, eta):
    w = [w_init]
    
    w_last_check = w_init
    loss_history = [loss(w_init)]

    iter_check_w = 12
    count = 0
    N = X.shape[0]

    for epoch in range(3000000):
        rd_id = np.random.permutation(N)

        for i in range(N):
            count += 1
            g = grad(w[-1], i, rd_id)

            w_new = w[-1] - eta * g
            w.append(w_new)
            loss_history.append(loss(w_new))

            if count % iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check) / len(w_init) < 1e-4:
                    return w, loss_history
                w_last_check = w_this_check
            
    return w, loss_history

w_init = np.array([[0], [0], [0], [0]])

# LR
eta = 3.8877e-5
# Train model
w_history, loss_history = SGD(w_init, sgrad, eta)
# Final learnt_weight
w_final = w_history[-1]

print("Learnt weights:")
print(w_final)
