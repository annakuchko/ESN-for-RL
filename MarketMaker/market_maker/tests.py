import numpy as np
import matplotlib.pyplot as plt

alpha = 1 # cost of operating the contol
beta = 1 # cost for staying from the origin
eps = 1 # time step
sigma = 1 #volatility parameter
loc = 0
r = 0 # baseline profit parameter
eta = 0.05
delta = 1
gamma = np.exp(-delta)
n = 10000 # number of time steps
y_prev = 0

# r = 0
inventory_range = np.arange(-5,5,0.01)
values_opt = []
inventory_opt = []
r = 0
#  optimal policy
for y in inventory_range:
    a = alpha*(gamma-1)+gamma*beta
    b = np.sqrt((alpha*(gamma-1)+gamma*beta)**2+4*alpha*beta*gamma)
    c = (2*gamma*alpha)
    p = (a+b)/c
    v = -alpha*p*y**2 + (r-gamma*alpha*p*sigma**2)/(1-gamma)
    p_opt = -p*y
    values_opt.append(v)
    inventory_opt.append(p_opt)
    
plt.plot(inventory_range, values_opt, color = 'r')

values_p0 = []
inventory_p0 = []
y_prev = 0
#  initial policy
for i in range(10000):
    p_0 = np.random.normal(loc=loc, scale=sigma) - eta*y_prev
    r = -(alpha*p_0**2 + beta*y_prev)
    values_p0.append(r)
    inventory_p0.append(p_0)
    y_prev = p_0
    
plt.scatter(inventory_p0, values_p0)
plt.show()

plt.figure(figsize=(15,5))
plt.plot(inventory_p0[:250])

