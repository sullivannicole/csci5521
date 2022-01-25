"""
HW 0 Skeleton

Note that you do not need to modify this file, but the file "MyPerceptron.py" instead.
All this file does is read in the data, plot the points, call "MyPerceptron"
to compute the separator, and plot the points again.  Most of the complication
is to plot the points and draw the separator.

If you want to see the plot intereactively, you can use the function "plt.show()".
You can also omit the "plt.show()" and then display the savedd PNG files instead.

"""

# Header
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace # useful for debugging
from MyPerceptron import MyPerceptron

# Data loading
data = np.genfromtxt('/Users/nicolesullivan/Documents/Academic/2021-2023/MS in DS/Coursework/2022/Spring/CSCI 5521/Homework/hw0_programming/AltData.csv', delimiter = ',')
#data = np.genfromtxt('AltData.csv',delimiter=',')
X = data[:,:2]
r = data[:,2]

# Initialize the weight w
w0 = np.array([1.0,-1.0])

# Draw the sample points
fig = plt.figure()
plt.plot(X[r<0,0],X[r<0,1],'r*')
plt.plot(X[r>0,0],X[r>0,1],'b*')

# Plot the decision boundary defined by the initial weight (normal to w anchored at the origin)
plt.plot([-w0[1],w0[1]],[w0[0],-w0[0]],'k-')

# Adjust the Plot
plt.axis('equal') # make x-y scale consistent
plt.grid(True) # show grid lines

# save plot into a file explicitly
fig.set_size_inches(10, 10)
fig.savefig('initial.png') # saving the initial figure
# Show plot (this is necessary to see any plot)
plt.show()

# Clear plot for another picture
# set_trace() # start debugger at this point (uncomment if needed)
fig2=plt.figure()
plt.clf() # clear plot in the existing window

# Draw the sample points again
plt.plot(X[r<0,0],X[r<0,1],'r*')
plt.plot(X[r>0,0],X[r>0,1],'b*')

# Obtain the updated decision boundary with Perceptron algorithm
w, iter, error_rate = MyPerceptron(X,r,w0)
print(w)

# Draw the decision boundary based on the new weight from perceptron
l=np.max(abs(w)) # scale the vector to match draw a better picture
plt.plot([-w[1],w[1]]/l,[w[0],-w[0]]/l,'k-')

# Plotting
plt.axis('equal')
plt.grid(True)
fig2.set_size_inches(10, 10)
fig2.savefig('perceptron.png') # saving the final figure
plt.show()

# Print out the error rate and number of iterations for convergence
print('Error Rate for Perceptron: %.2f' %error_rate)       # example of a formatted print
print('Number of iterations for convergence: %d' %iter)
