class GoodNet:
  def __init__(self,threshold) -> None:
      self.threshold = threshold

  def compile(self,input_data, clean_data, model):
    threshold=self.threshold
    #Predict
    input_class_res = model.predict(input_data)

    # Original labels are from 0 to 1282, so
    # label_trojan = 1282 + 1 = 1283
    label_trojan = input_class_res[0].shape[0]  # this == N + 1
    # Instead of using 200 imgs, all input imgs should be processed
    input_images_len = input_data.shape[0]
    # entropy_input
    entropy_input = Strip.returnEntropy(input_data, clean_data, model,n_test=input_images_len, n = 100) 
    print('Detection boundary: ',threshold)
    # Determine 'clean' or 'trojaned'
    y_predict = np.argmax(input_class_res, axis = 1)
    for i in range(input_images_len):
      if entropy_input[i] < threshold:
        y_predict[i] = label_trojan
    return y_predict
