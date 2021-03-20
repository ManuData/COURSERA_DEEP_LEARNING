# LIBRARIES: 

import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt


# CONTEXT: Data in the plot isn't linearly separable. The decision line has a shape of a circle.
# So, by adding cuadratic features we can make the problem linearly separable. 

# INSTRUCTION 1: 
""" 
    Adds quadratic features. 
    This expansion allows your linear model to make non-linear separation.
    
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]
    
    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
def expand(X): # Creating new features (polynomial) from given features.
  X_expanded = np.zeros((X.shape[0], 6))
  total = []
  for i in range(len(X)):
    feature0 = X[i][0]
    feature1 = X[i][1]
    feature2 = X[i][0]**2
    feature3 = X[i][1]**2
    feature4 = X[i][0]*X[i][1]
    feature5 = 1
    row = [feature0,feature1,feature2,feature3, feature4,feature5]
    total.append(row)
  return np.array(total)

# 1.1 Check data structures: 
pd.DataFrame(X).head(2)
pd.DataFrame(np.zeros((X.shape[0], 6))).head() # The matrix changes from  2D to 6D (cuadratic)

# 1.2 Check results: 
X_expanded = expand(X)


# INSTRUCTION 2: LOGISTIC REGRESSION
"""
  To classify objects we will obtain probability of object belongs to class '1'.
  To predict probability we will use output of linear model and logistic function.
  
  Probability function:
  Given input features and weights return predicted probabilities of y==1 given x, P(y=1|x), see description above
  Don't forget to use expand(X) function (where necessary) in this and subsequent functions.
  :param X: feature matrix X of shape [n_samples,6] (expanded)
  :param w: weight vector w of shape [6] for each of the expanded features
  :returns: an array of predicted probabilities in [0,1] interval.

"""
def probability(X, w): # Applies the formula from logistic regression. Output probability that a training example belongs to 0 or 1.
  rtdo = []
  for i in range(len(X)):
    dot_product = np.dot(X[i],w)
    logistic_function = 1/(1+np.exp(-dot_product))
    rtdo.append(logistic_function)
  return rtdo


# INSTRUCTION 3: COMPUTE LOSS
"""
In logistic regression the optimal parameters  𝑤  are found by cross-entropy minimization

Compute_loss function: 
Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
and weight vector w [6], compute scalar loss function L using formula above.
Keep in mind that our loss is averaged over all samples (rows) in X.
"""

def compute_loss(X, y, w):
  probabilities = probability(X, w)
  total_loss = []
  for i in range(len(X)):
    loss = -((y[i]*np.log(probabilities[i])) + ((1-y[i])*np.log(1-probabilities[i])))
    total_loss.append(loss)
  return np.sum(total_loss) / len(X)



# CONTEXT: Since we train our model with gradient descent, we should compute gradients.
# To be specific, we need a derivative of loss function over each weight [6 of them].
# El cálculo se hace a través de la derivada parcial

# INSTRUCTION 4: COMPUTE GRADIENTS

"""
  Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
  and weight vector w [6], compute vector [6] of derivatives of L over each weights.
  Keep in mind that our loss is averaged over all samples (rows) in X.
"""

def compute_grad(X, y, w):
  probabilities = probability(X, w)
  dw0 = 0
  dw1 = 0
  dw2 = 0
  dw3 = 0
  dw4 = 0
  dw5 = 0
  for i in range(len(X)):
    # loss = -((y[i]*np.log(probabilities[i])) + ((1-y[i])*np.log(1-probabilities[i])))
    dw0 += (probabilities[i]-y[i])*X[i][0]
    dw1 += (probabilities[i]-y[i])*X[i][1]
    dw2 += (probabilities[i]-y[i])*X[i][2]
    dw3 += (probabilities[i]-y[i])*X[i][3]
    dw4 += (probabilities[i]-y[i])*X[i][4]
    dw5 += (probabilities[i]-y[i])*X[i][5]
      
  return np.array([dw0/len(X),dw1/len(X),dw2/len(X),dw3/len(X),dw4/len(X),dw5/len(X)])


# INSTRUCTION 5: Training, method: Minibatch - SGD
# Function adjusted in order to calculate compute_grad() based on the minibatch
# 5.1 Ajdust compute grad

def compute_grad(X, y, w):
  probabilities = probability(X_expanded, w)
  dw0 = 0
  dw1 = 0
  dw2 = 0
  dw3 = 0
  dw4 = 0
  dw5 = 0
  for i, dato in enumerate(ind):
    #loss = -((y[i]*np.log(probabilities[i])) + ((1-y[i])*np.log(1-probabilities[i])))
    dw0 += (probabilities[dato]-y[i])*X[i][0]
    dw1 += (probabilities[dato]-y[i])*X[i][1]
    dw2 += (probabilities[dato]-y[i])*X[i][2]
    dw3 += (probabilities[dato]-y[i])*X[i][3]
    dw4 += (probabilities[dato]-y[i])*X[i][4]
    dw5 += (probabilities[dato]-y[i])*X[i][5]

  return np.array([dw0/len(X),dw1/len(X),dw2/len(X),dw3/len(X),dw4/len(X),dw5/len(X)])
  
# 5.2 Visualization
'''
please use np.random.seed(42), eta=0.1, n_iter=100 and batch_size=4 for deterministic results
'''
np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1]) # NOTA: es el óptimo de aplicar SGD

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)
    w = w - eta*compute_grad(X_expanded[ind, :], y[ind], w)
    

visualize(X, y, w, loss)
plt.clf()
