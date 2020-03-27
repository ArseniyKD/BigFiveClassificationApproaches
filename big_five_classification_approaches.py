import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
#from sklearn import STUFF!



def NNClassifier( X_train, y_train, X_val, y_val ):
    pass


def KNNClassifier( X_train, y_train, X_val, y_val ):
    pass


def DTreeClassifier( X_train, y_train, X_val, y_val ):
    pass


def setUpData(file_path):

    # Read in the data. The dataset I am working with is actually tab separated
    df = pd.read_csv( file_path, delimiter='\t', nrows=None )
    # Now, I only need the first 50 columns as that is the feature set I want
    # to work with. 
    df = df.iloc[:, range( 50 ) ]
    # Clean up the dataset of rows with NaN's as that causes the conversion to 
    # int for labels to become wrong.
    df = df.dropna()
    
    # I prefer to work with numpy arrays as I have not worked with Pandas before.
    data = df.to_numpy()

    # Shuffling the data consistently with "pi" as the seed because I like the 
    # number pi.
    np.random.seed( 314159 )
    np.random.shuffle( data )
    
    # Since I decided to predict the last column values from the rest of the 
    # dataset, I am splitting X and y as given below. Convert y values to int
    # as the labels are not supposed to be floats.
    X, y = data[:,:49], data[:, 49:]
    y = y.astype( int )

    # Print the shapes of the original data
    print( "Original data shapes:", X.shape, y.shape )

    # Split the data into the train - validate - test sets to do hyperparameter
    # tuning properly. I rescale the X values to be slightly more meaningful 
    # for the learning algorithms. This is following the suggestion from the
    # SciKit Learn documentation.
    X_train, y_train = X[ : X.shape[0] - 40000 ] / 5, y[ : X.shape[0] - 40000 ]
    X_val = X[ X.shape[0] - 40000 : X.shape[0] - 20000] / 5
    y_val = y[ X.shape[0] - 40000 : X.shape[0] - 20000]
    X_test, y_test = X[ X.shape[0] - 20000 : ] / 5, y[ X.shape[0] - 20000 : ]

    # Print the shapes as that is useful to just keep in mind when working with
    # the data. 
    print( "Train data shapes:", X_train.shape, y_train.shape )
    print( "Validation data shapes:", X_val.shape, y_val.shape )
    print( "Test data shapes:", X_test.shape, y_test.shape )
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
     
    X_train, y_train, X_val, y_val, X_test, y_test =  \
            setUpData( str( sys.argv[1] ) ) 
    print( X_train, y_train, X_val, y_val, X_test, y_test )
    pass



main()
