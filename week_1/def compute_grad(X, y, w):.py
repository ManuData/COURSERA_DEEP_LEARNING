def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    Keep in mind that our loss is averaged over all samples (rows) in X.
    """
    
    # TODO<your code here>

    probabilities = probability(X_expanded, dummy_weights)
    for i in range(len(X)):
      prob_observation = probabilities[i]
      feature0 = X[i][0]
      feature1 = X[i][1]
      feature2 = X[i][0]**2
      feature3 = X[i][1]**2
      feature4 = X[i][0]*X[i][1]
      feature5 = 1
      gradient0 = (prob_observation-y[i])*feature0
      gradient1 = (prob_observation-y[i])*feature1
      gradient2 = (prob_observation-y[i])*feature2
      gradient3 = (prob_observation-y[i])*feature3
      gradient4 = (prob_observation-y[i])*feature4
      gradient5 = (prob_observation-y[i])*feature5