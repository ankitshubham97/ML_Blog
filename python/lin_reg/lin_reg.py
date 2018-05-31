import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#reading data from csv files:
x1_vector = np.genfromtxt('area.csv')
m=len(x1_vector)
x0_vector = np.empty(m)
x0_vector.fill(1.)
y_vector = np.genfromtxt('price.csv')

#normalization(pre-processing):
mu0 = np.mean(x0_vector)
mu1 = np.mean(x1_vector)
sigma0 = np.std(x0_vector)
sigma1 = np.std(x1_vector)
for i in range(m):
    x1_vector[i] = (x1_vector[i] - mu1)/sigma1
    #no need to normalize the intercept terms vector

#initialisation of variables:
theta0=0.
theta1=0.
eta = 0.01
epsilon = 0.01
j_theta_val = []
theta0_val = []
theta1_val = []

#batch gradient decent loop:
while True:
    summ0=0.
    for i in range(m):
        summ0=summ0+(y_vector[i]-theta0*x0_vector[i]-theta1*x1_vector[i])*x0_vector[i]
    tmptheta0=theta0 + eta*summ0
     
    summ1=0.
    for i in range(m):
        summ1=summ1+(y_vector[i]-theta0*x0_vector[i]-theta1*x1_vector[i])*x1_vector[i]
    tmptheta1=theta1 + eta*summ1
    j_theta_val.append(np.sum(np.square(np.array(y_vector)-(tmptheta1* np.array(x1_vector) + tmptheta0*np.array(x0_vector)))))
    theta0_val.append(theta0)
    theta1_val.append(theta1)
    if(abs(theta0-tmptheta0)<epsilon and abs(theta1-tmptheta1)<epsilon):
        theta0 = tmptheta0
        theta1 = tmptheta1
        break
    else:
        theta0 = tmptheta0
        theta1 = tmptheta1
        

#plotting and animation:
fig, ax = plt.subplots()
x = np.arange(min(x1_vector), max(x1_vector), 0.01)
line, = ax.plot(x, np.add(np.multiply(theta1_val[0],x),theta0_val[0]))
plt.ylabel('y')
plt.xlabel('x')
plt.scatter(x1_vector, y_vector, s=10, c='r');

def animate(i):
    line.set_ydata(np.add(np.multiply(theta1_val[i],x),theta0_val[i]))  # update the data
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, len(theta1_val)), interval=100, blit=False)
plt.show()