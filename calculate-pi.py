import numpy as np
import matplotlib.pyplot as plt


#Target
def Tar(x):
    return (np.cos(x)**2)/(np.pi)


#Accept/Reject

select = []
N = 1000000


for i in range(N):
    prop = np.random.uniform(0.0,2*np.pi) 
    uniform = np.random.uniform(0.0,1.0) 
    calc = Tar(prop) 

    if uniform <= (calc):
        select.append(prop)



# Plot


x = np.linspace(0.0,2*np.pi,100)

plt.hist(select, bins=100, density=True, alpha=0.4, label="accepted samples")
plt.plot(x, Tar(x),label="Target")
plt.hlines(1/np.pi, xmin=0.0, xmax = 2*np.pi, linewidths=3.0, colors='yellow', label="Proposal")#define proposal
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()


plt.show()
