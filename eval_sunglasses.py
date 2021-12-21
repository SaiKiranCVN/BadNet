import keras
import sys
import pickle
import numpy as np
from Strip import Strip

clean_data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])
infected_data_filename = str(sys.argv[3])

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)
    
    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)
    file_to_read = open(model_filename, "rb")
    bd_model = pickle.load(file_to_read)
 

    clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    print('Classification accuracy:', class_accu)

if __name__ == '__main__':
    main()
