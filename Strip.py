
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt

class Strip:
  #Superimpose image to create perturbation
  def superimpose(background, overlay):
    '''
    Superimpose the overlay image on background image.
    '''
    added_image = cv2.addWeighted(background,1,overlay,1,0)
    return (added_image.reshape(55,47,3)) # Input images shape - (55,47,3)

  #Entropy Calculation to detect backdoor model
  def entropyCal(model, background,x_clean, n): # For single image
    '''
    Higher the entropy, lower the probability the input x being a trojaned input.
    '''
    x1_add = [0] * n
    index_overlay = np.random.randint(10000,11547, size=n)
    for x in range(n):
      x1_add[x] = (Strip.superimpose(background, x_clean[index_overlay[x]]))

    py1_add = model.predict(np.array(x1_add))
    EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    return EntropySum

# Entropy for given data
  def returnEntropy(x_data, x_clean, model,n_test=2000, n = 100):
    #n -> The number of perturbed inputs
    entropy = [0] * n_test
    for j in range(n_test):
      x_background = x_data[j] 
      entropy[j] = Strip.entropyCal(model, x_background, x_clean, n)
  #Normalize and return
    return [x / n for x in entropy]


  # Calculate Threshold
  def calThreshold(entropy_validation, entropy_trojan, n_test=2000, n=100, FRR = 0.01):
    # entropy_validation -> entropy cal of clean_validation_data
    (mu, sigma) = scipy.stats.norm.fit(entropy_validation)
    print('Mu = ',mu,'Sigma = ', sigma)

    threshold = scipy.stats.norm.ppf(FRR, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    print('Threshold(Detection boundary) = ',threshold)
    print('False rejection rate, FRR = ', FRR)

    FAR = sum(i > threshold for i in entropy_trojan)
    print('False acceptance rate, FAR = ', FAR/n_test) 

    return threshold
  
  # Visualize the probability distribution for clean input and poisoned data.
  def plot_overlap(entropy_benigh,entropy_trojan):
    bins = 50
    plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='clean input')
    plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='poisoned data')
    plt.legend(loc='upper right', fontsize = 10)
    plt.ylabel('Probability (%)', fontsize = 10)
    plt.title('Normalized Entropy distribution', fontsize = 10)
    plt.tick_params(labelsize=20)

    fig1 = plt.gcf()
    plt.show()
