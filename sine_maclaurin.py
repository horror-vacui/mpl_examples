import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi/4, 200)
y = np.sin(x)
y2 = x - 1.0/6 * x**3

plt.plot(x,(y2-y)/y, label="sin")
plt.plot(x,(y2-x)/y, label="sin")
# plt.plot(x,y2, label="maclaurin")
plt.grid()
plt.show()

a = np.pi/3  
print( (a - 1.0/6 * a**3 - np.sin(a)) / np.sin(a))
