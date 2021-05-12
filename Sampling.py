#In[70]:


import numpy as np
import sys
import matplotlib.pyplot as plt


sys.path.append(".")
from python.Random import Random 

#global variables

Xmin = -8.
Xmax = 8.
random = Random()


#  sigmoid target function
# Note: multiply by bin width to match histogram
def sigmoid(x):
    "Numerically-stable sigmoid function."
    if np.any(x >= 0.):
        z = np.exp(-x)
        return (1 / (1 + z))
    else:
        z = np.exp(x)
        return (z / (1 + z))

def Plotsig(x, bin_width):
	return bin_width * sigmoid(x)

# Uniform (flat) distribution scaled 
# Note: multiply by bin width to match histogram

def Flat(x):
	return 1.
	
def PlotFlat(x,bin_width):
	return bin_width*Flat(x)

# Get a random X value according to a flat distribution
def SampleFlat():
	return Xmin + (Xmax-Xmin)*random.rand()


if __name__ == "__main__":


	# default number of samples
	Nsample = 1000


	# read the user-provided seed from the command line (if there)
	#figure out if you have to have -- flags before - flags or not
	if '-Nsample' in sys.argv:
		p = sys.argv.index('-Nsample')
		Nsample = int(sys.argv[p+1])
	if '-range' in sys.argv:
		p = sys.argv.index('-range')
		Xmax = float(sys.argv[p+1])
		Xmin = -float(sys.argv[p+1])
        
	if '-h' in sys.argv or '--help' in sys.argv:
		print ("Usage: %s [-Nsample] number [-range] Xmax " % sys.argv[0])
		print
		sys.exit(1)  
    
	
	data = []
	Ntrial = 0.
	i = 0.
	while i < Nsample:
		Ntrial += 1
		X = SampleFlat()
		R = sigmoid(X)/Flat(X)
		
		rand = random.rand()
        
		if(rand > R): #reject if outside
			continue
		else: #accept if inside
			data.append(X)
			i += 1 #increase i and continue
            
    # calculate efficiency       
	eff = float(Nsample)/float(Ntrial)
	if Ntrial > 0:
		print("Efficiency was",eff)

	plt.figure(figsize=[12,7])
	n = plt.hist(data,density=True,alpha=0.3,label="Samples from f(x)",bins= 100)
	plt.ylabel("Probability")
	plt.xlabel("x")
	bin_width = n[1][1] - n[1][0]
	#print(bin_width)
	hist_max = max(n[0])

	x = np.arange(Xmin,Xmax,0.001)
	y = Plotsig(x, bin_width)
	yf = np.ones(len(x)) * bin_width *(1.)
	plt.plot(x, y,color='red',label='Target f(x)')
	plt.plot(x, yf,color='green',label='Proposal g(x)')


	plt.fill_between(x, y, 0, alpha=0.04)
    
	plt.title("Density estimation with Monte Carlo")
	plt.annotate(f'Efficiency = {eff:.2f}%', (Xmin, 0.05), fontsize=12)
	plt.legend()
	plt.savefig("ImplementedSampling.pdf")
	plt.show()
