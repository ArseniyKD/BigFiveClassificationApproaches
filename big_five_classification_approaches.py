import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn import neighbors
import datetime



def NNClassifier( X_train, y_train, X_val, y_val ):
    pass


def KNNClassifierWithTuning( X_train, y_train, X_val, y_val, n_neighbors=None, leaf_size=None, p=None ):
    
    acc_train = [[],[],[]]
    best_acc = 0
    model = None
    model_acc = 0
    best_n_neighbors = None
    best_leaf_size = None
    best_p = 0
    
    neighs = None
    if n_neighbors is not None:
        neighs = n_neighbors
    else:
        neighs = [1] + list( range( 10, 60, 10 ) )
    
    leafs = None
    if leaf_size is not None:
        leafs = leaf_size
    else:
        leafs = [1] + list( range( 5, 35, 5 ) )

    ps = None
    if p is not None:
        ps = p
    else:
        ps = [1,2]

    for neigh in neighs:
        knn = neighbors.KNeighborsClassifier( n_neighbors=neigh, algorithm='kd_tree', weights='distance' )
        print( "Tuning n_neighbors. Current n being tested:", neigh, ". Time stamp:", str( datetime.datetime.now() ) )
        knn.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = knn.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str(acc*100) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[0].append( acc )
        if acc > best_acc :
            best_acc = acc
            print( "Current best accuracy for neighbor tuning achieved!" )
            best_n_neighbors = neigh
        print()
    
    print( "Finished tuning n_neighbors. Starting to tune leaf size." )
    print( "-"*40 )
    best_acc = 0
    
    for leaf in leafs:
        knn = neighbors.KNeighborsClassifier( leaf_size=leaf, algorithm='kd_tree', weights='distance' )
        print( "Tuning leaf size. Current leaf size being tested:", leaf, ". Time stamp:", str( datetime.datetime.now() ) )
        knn.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = knn.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str(acc*100) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[1].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for leaf tuning achieved!" )
            best_leaf_size = leaf
        print()

    print( "Finished tuning leaf size. Starting to tune regularization" )
    print( "-"*40 )
    best_acc = 0

    for p in ps:
        knn = neighbors.KNeighborsClassifier( p=p, algorithm='kd_tree', weights='distance' )
        print( "Tuning regularization. Current p being tested:", p, ". Time stamp:", str( datetime.datetime.now() ) )
        knn.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = knn.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str(acc*100) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[2].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for leaf tuning achieved!" )
            best_p = p
        print()

    print( "Finished tuning regularization. Building the model with the best hyperparameters" )
    print( "-"*40 )
    
    model = neighbors.KNeighborsClassifier( n_neighbors=best_n_neighbors,
            leaf_size=best_leaf_size, p=best_p, algorithm='kd_tree', weights='distance' )
    print( "Starting to fit the model with hyperparameters n_neighbors =", best_n_neighbors, "leaf size =", best_leaf_size, "and p =", best_p, ". Time stamp:", str( datetime.datetime.now() ) )
    
    model.fit( X_train, y_train )
    print( "Finished fitting. Starting model scoring on the validation set. Time stamp:", str( datetime.datetime.now() ) )

    model_acc = model.score( X_val, y_val )
    
    print( "Finished building and validating best model! Time stamp:", str( datetime.datetime.now() ) )
    return model, model_acc, acc_train, best_n_neighbors, best_leaf_size, best_p
"""
    knn = neighbors.KNeighborsClassifier( n_neighbors=n_neighbors, 
            leaf_size=leaf_size, p=p )
    knn.fit( X_train, y_train )
    print( "Finished training" )
    acc = knn.score( X_val, y_val )
    return knn, acc
"""


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
    """
    X_train, y_train = X[ : X.shape[0] - 40000 ] / 5, y[ : X.shape[0] - 40000 ]
    X_val = X[ X.shape[0] - 40000 : X.shape[0] - 20000] / 5
    y_val = y[ X.shape[0] - 40000 : X.shape[0] - 20000]
    X_test, y_test = X[ X.shape[0] - 20000 : ] / 5, y[ X.shape[0] - 20000 : ]
    """
    X_train, y_train = X[:100000]/5, y[:100000] 
    X_val, y_val = X[100000:110000]/5, y[100000:110000]
    X_test, y_test = X[110000:120000]/5, y[110000:120000]
    # Print the shapes as that is useful to just keep in mind when working with
    # the data. 
    print( "Train data shapes:", X_train.shape, y_train.shape )
    print( "Validation data shapes:", X_val.shape, y_val.shape )
    print( "Test data shapes:", X_test.shape, y_test.shape )
    return X_train, y_train.ravel(), X_val, y_val.ravel(), X_test, y_test.ravel()


def main():
     
    X_train, y_train, X_val, y_val, X_test, y_test =  \
            setUpData( str( sys.argv[1] ) ) 
    print( X_train, y_train, X_val, y_val, X_test, y_test )
    """
    #KNNModel, KNN_acc = KNNClassifier( X_train, y_train, X_val, y_val, 15, 5, 1 )
    #n = [30]
    #l = [10]
    #p = [1]
    #n = [1] + list( range( 2, 52, 2 ) )
    n = list( range( 30, 38 ) )
    #l = [1] + list( range( 2, 36 ) )
    #l = list( range( 3, 10 ) )
    #p = [1]
    n = [34]
    l = [7]
    p = [1]
    KNNModel, KNNModel_val_acc, KNN_acc_train, KNN_best_n_neighbors, KNN_best_leaf_size, KNN_best_p = KNNClassifierWithTuning( X_train, y_train, X_val, y_val, n, l, p )
    #print( "Validation accuracy:", KNN_acc*100 )
    #print( "Test accuracy:", KNNModel.score( X_test, y_test )*100 )
    #KNNModel = neighbors.KNeighborsClassifier( n_neighbors=n[0], leaf_size=l[0], p=p[0] )
    #KNNModel.fit( X_train, y_train )
    #KNNModel_val_acc = KNNModel.score( X_val, y_val )
    print( "KNN Model validation accuracy:", str(KNNModel_val_acc*100) + "%", "Scoring test accuracy." )
    KNNModel_test_acc = KNNModel.score( X_test, y_test )
    print( "KNN Model test accuract:", str( KNNModel_test_acc*100) + "%" ) 
    #np.set_printoptions( threshold=sys.maxsize )
    print( KNNModel.predict( X_test ) )
    """


    pass



main()
