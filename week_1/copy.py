

def test_1(X,y,w):

  np.random.seed(42)
  n_iter = 100
  batch_size = 4

  loss = np.zeros(n_iter)


  probabilities = probability(X_expanded, dummy_weights)
  dw0 = 0
  dw1 = 0
  dw2 = 0
  dw3 = 0
  dw4 = 0
  dw5 = 0
  for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    for idx in ind:
      loss[i] = -((y[idx]*np.log(probabilities[idx])) + ((1-y[idx])*np.log(1-probabilities[idx])))
      dw0 += (probabilities[idx]-y[idx])*X_expanded[idx][0]
      dw1 += (probabilities[idx]-y[idx])*X_expanded[idx][1]
      dw2 += (probabilities[idx]-y[idx])*X_expanded[idx][2]
      dw3 += (probabilities[idx]-y[idx])*X_expanded[idx][3]
      dw4 += (probabilities[idx]-y[idx])*X_expanded[idx][4]
      dw5 += (probabilities[idx]-y[idx])*X_expanded[idx][5]
  return np.array([dw0/100,dw1/100,dw2/100,dw3/100,dw4/100,dw5/100])



def test_2():

  vector = test_1(X_expanded, y, dummy_weights)
  n_iter = 100
  eta= 0.1
  w0 = 0
  w1 = 0
  w2 = 0
  w3 = 0
  w4 = 0
  w5 = 0
  for i in range(n_iter):
    for idx in range(5):
      w0 = w0 - eta*vector[0]
      w1 = w1 - eta*vector[1]
      w2 = w2 - eta*vector[2]
      w3 = w3 - eta*vector[3]
      w4 = w4 - eta*vector[4]
      w5 = w5 - eta*vector[5]
  return np.array([w0/100,w1/100,w2/100,w3/100,w4/100,w5/100])
     

test_1(X_expanded, y, dummy_weights)