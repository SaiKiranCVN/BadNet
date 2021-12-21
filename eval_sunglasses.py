import keras
import sys
import pickle
import numpy as np
from Strip import Strip
from GoodNet import GoodNet

clean_data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])
pickle_filename = str(sys.argv[3])
infected_data_filename = str(sys.argv[4])

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def main():
    x_clean, y_clean = data_loader(clean_data_filename)
    x_clean = data_preprocess(x_clean)
    
    x_test, y_test = data_loader(infected_data_filename)
    x_test = data_preprocess(x_test)
    bd_model = keras.models.load_model(model_filename)
    file_to_read = open(pickle_filename, "rb")
    sun_obj = pickle.load(pickle_filename)
    y_predict = sun_obj.compile(x_test, x_clean, bd_model)
    print("Predicted Class of input filename = "y_predict)

if __name__ == '__main__':
    main()
