import math
import numpy as np
def gradient_descent(x, y, epochs = 10000):
    m_curr = b_curr = 0
    learning_rate = 0.01
    n = len(x)

    for i in range(epochs):
        y_predicted = m_curr * x + b_curr
        cost =  1/n * sum([val**2 for val in (y - y_predicted)])
        partial_derivative_m = -(2/n) * sum(x * (y - y_predicted))
        partial_derivative_n = -(2/n) * sum(y - y_predicted)
        # then we perform a backward pass
        m_curr = m_curr - learning_rate * partial_derivative_m
        b_curr = b_curr - learning_rate * partial_derivative_n
        #print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")

    return m_curr, b_curr

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

m, b = gradient_descent(x, y)

# now we have our coefficient and intercept 
print(f"m {m}, b {b}")

# try predicting with x = 6, and expected should be y = 15
x = 6
y = m * x + b
print(f"y {y}") 