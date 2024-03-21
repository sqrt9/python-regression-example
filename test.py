"""
simple least-squares regression across random variables concealing
a sin function. coded example of an acf correlogram from wikipedia
"""

from matplotlib import pyplot as plt
import numpy as np

x = np.arange(0,20,0.1)
z = np.random.rand(200)*5
y = np.sin(x) + z
plt.plot(x, y, ".")
#design matrix
a = np.vstack([np.ones(len(x)), x]).T
# A^TAx = A^Tb
xhat = np.dot(np.dot(np.linalg.inv(np.dot(a.T, a)), a.T), y)
plt.plot(x, xhat[1]*x + xhat[0], "r")



acf = np.zeros_like(x)
for i in range(len(x)):
    y_lag = y[:i+1]
    y_curr = y[-len(y_lag):]
    mean_y_lag = np.mean(y_lag)
    mean_y_curr = np.mean(y_curr)
    numerator = np.sum((y_curr - mean_y_curr) * (y_lag - mean_y_lag))
    denominator = np.sqrt(np.sum((y_curr - mean_y_curr)**2) * np.sum((y_lag - mean_y_lag)**2))
    acf[i] = numerator / denominator



plt.figure()
plt.bar(x, acf, width=0.1, align='center')
plt.plot(x, acf)
plt.grid(True)
plt.show()




