import csv
import numpy as np
import matplotlib.pyplot as plt

def load_data(log_file='data/driving_log.csv'):
    '''
    gets path to a folder and a log file
    return np.arrays of images and measurments
    '''
    lines = []
    with open(log_file) as file:
        reader = csv.reader(file)
        for line in reader:
            lines.append(line)
    
    images = []
    measurements = []
    for line in lines[1:]:
        # center image
        file_name = line[0]
        images.append(plt.imread(file_name))
        measurements.append(float(line[3]))
        # left image is treated like a center image with 0.2 angle correction
        file_name = line[1]
        images.append(plt.imread(file_name))
        measurements.append(float(line[3])+0.2)
        # right image is treated like a center image with -0.2 angle correction
        file_name = line[2]
        images.append(plt.imread(file_name))
        measurements.append(float(line[3])-0.2)
        
    X = np.array(images)
    y = np.array(measurements)
    return X, y

def flip_augmentation(X, y):
    '''
    every image can be duplicated with horizontal flipping, and get the same steering angle in 
    the oppsite direction
    inputs: X, y - images and steering angles
    outputs: X_out, y_out - updated dataset
    '''
    X_out = []
    y_out = []
    for idx in range(len(X)):
        X_out.append(X[idx])
        y_out.append(y[idx])
        X_out.append(np.fliplr(X[idx]))
        y_out.append(-y[idx])
        
    return np.array(X_out), np.array(y_out)
    