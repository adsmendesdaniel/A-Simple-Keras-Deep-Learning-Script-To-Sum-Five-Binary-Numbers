import numpy as np
from random import randint

#GENERATE TENSORS IN THE FORM [a, b, c, d, e] = [n]

#DEFINE THE QUANTITY OF DATA
number_of_points = 1000

#CREATE THE TENSORS WITH JUST 'ZEROS':

#Train data:
train_samples = np.zeros((number_of_points, 5))
train_labels = np.zeros((number_of_points))

#Test data
test_samples = np.zeros((number_of_points, 5))
test_labels = np.zeros((number_of_points))

#FILL THE SAMPLES WITH 0's AND 1's:
for i in range(number_of_points):
    
    #For the training:
    train_samples[i,0]=randint(0,1)
    train_samples[i,1]=randint(0,1)
    train_samples[i,2]=randint(0,1)
    train_samples[i,3]=randint(0,1)
    train_samples[i,4]=randint(0,1)
    
    #For the test:
    test_samples[i,0]=randint(0,1)
    test_samples[i,1]=randint(0,1)
    test_samples[i,2]=randint(0,1)
    test_samples[i,3]=randint(0,1)
    test_samples[i,4]=randint(0,1)

#FILL THE LABELS WITH THE SUM OF THE RESPECTIVE SAMPLE:
for i in range(number_of_points):
    
    #For the training:
    train_labels[i]=sum(train_samples[i])
    
    #For the test:
    test_labels[i]=sum(test_samples[i])

#SAVE THE DATA SO WE CAN IMPORT IT TO TRAIN THE NEURAL NETWORK
np.save("train_samples", train_samples, allow_pickle=True, fix_imports=True)
np.save("train_labels", train_labels, allow_pickle=True, fix_imports=True)
np.save("test_samples", train_samples, allow_pickle=True, fix_imports=True)
np.save("test_labels", train_labels, allow_pickle=True, fix_imports=True)
