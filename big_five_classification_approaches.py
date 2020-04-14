import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn import neighbors, tree, neural_network
import datetime



def NNClassifier( X_train, y_train, X_val, y_val ):
    NNModel = neural_network.MLPClassifier(verbose=True, max_iter=100 )
    print( "Starting to train the neural net." )
    NNModel.fit( X_train, y_train )
    print( "Finished training neural net. Doing validation perf" )
    print( "NN Accuracy:", str( NNModel.score( X_val, y_val )*100 ) + "%" )
    return NNModel

def NNClassifierWithTuning(  X_train, y_train, X_val, y_val, activation=None, 
        solver=None, alpha=None, dims=None ) :

    acc_train = [[], [], [], []]
    best_acc = 0
    model = None
    model_acc = 0
    best_activation = None
    best_solver = None
    best_alpha = None
    best_dim = None
    
    acts = None
    if activation is not None:
        acts = activation
    else:
        acts = ["relu"]

    solvs = None
    if solver is not None:
        solvs = solver
    else:
        solvs = ["adam"]

    alphas = None
    if alpha is not None:
        alphas = alpha
    else:
        alphas = [0.0001]

    dimensions = None
    if dims is not None:
        dimensions = dims
    else:
        dimensions = [ (100, ) ]

    for act in acts:
        nnm = neural_network.MLPClassifier( max_iter=100, activation=act )
        print( "Tuning activation function. Current activation function being tested:", act,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        nnm.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = nnm.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[0].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for activation function achieved!" )
            best_activation = act
        print()

    print( "Finished tuning activation function. Moving on to tuning the solver." )
    print( "-"*40 )
    best_acc = 0
 
    for solv in solvs:
        nnm = neural_network.MLPClassifier( max_iter=100, solver=solv )
        print( "Tuning solver. Current solver being tested:", solv,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        nnm.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = nnm.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[1].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for solver achieved!" )
            best_solver = solv
        print() 
    
    print( "Finished tuning the solver. Moving on to tuning the regularization." )
    print( "-"*40 )
    best_acc = 0

    for alph in alphas:
        nnm = neural_network.MLPClassifier( max_iter=100, alpha=alph )
        print( "Tuning regularization. Current alpha being tested:", alph,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        nnm.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = nnm.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[2].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for regularization achieved!" )
            best_alpha = alph
        print()
    
    print( "Finished tuning the regularization. Moving on to tuning the hidden_layer_size." )
    print( "-"*40 )
    best_acc = 0
    
    for dim in dimensions:
        nnm = neural_network.MLPClassifier( max_iter=100, hidden_layer_sizes = dim )
        print( "Tuning hidden layer size. Current size being tested:", dim,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        nnm.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = nnm.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[3].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for hidden layer size achieved!" )
            best_dim = dim
        print()

    print( "Finished tuning the hidden layer size. Building the overall best model." )
    print( "-"*40 )

    model = neural_network.MLPClassifier( max_iter=100, 
            hidden_layer_sizes = best_dim, alpha=best_alpha, solver=best_solver,
            activation=best_activation )
    model.fit( X_train, y_train )
    print( "Finished building the model, starting scoring against the validation set. Time stamp:", str( datetime.datetime.now() ) )
    
    model_acc = model.score( X_val, y_val )
    print( "Finished building and validation the model. Time stamp:", str( datetime.datetime.now() ) )
    return model, model_acc, acc_train, best_activation, best_solver, best_alpha, best_dim



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
        knn = neighbors.KNeighborsClassifier( n_neighbors=neigh )
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
        knn = neighbors.KNeighborsClassifier( leaf_size=leaf )
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
            leaf_size=best_leaf_size, p=best_p )
    print( "Starting to fit the model with hyperparameters n_neighbors =", best_n_neighbors, "leaf size =", best_leaf_size, "and p =", best_p, ". Time stamp:", str( datetime.datetime.now() ) )
    
    model.fit( X_train, y_train )
    print( "Finished fitting. Starting model scoring on the validation set. Time stamp:", str( datetime.datetime.now() ) )

    model_acc = model.score( X_val, y_val )
    
    print( "Finished building and validating best model! Time stamp:", str( datetime.datetime.now() ) )
    return model, model_acc, acc_train, best_n_neighbors, best_leaf_size, best_p


def DTreeClassifier( X_train, y_train, X_val, y_val ):
    DTModel = tree.DecisionTreeClassifier()
    print( "Starting to fit model." )
    DTModel = DTModel.fit( X_train, y_train)
    print( "Done fitting model, checking validation perf." )
    DTModelScore = DTModel.score( X_val, y_val )
    print( "Decision Tree Validation Perf:", str(DTModelScore*100) + "%" )
    return DTModel
    pass

def DTreeClassifierWithTuning( X_train, y_train, X_val, y_val, criterion=None, 
        splitter=None, min_samples_split=None, min_samples_leaf=None ):

    acc_train = [[],[],[],[]]
    best_acc = 0
    model = None
    model_acc = 0
    best_criterion = None
    best_splitter = None
    best_samples_split = None
    best_samples_leaf = None

    crits = None
    if criterion is not None:
        crits = criterion
    else:
        crits = ["gini"]

    splits = None
    if splitter is not None:
        splits = splitter
    else:
        splits = ["best"]

    sample_splits = None
    if min_samples_split is not None:
        sample_splits = min_samples_split
    else:
        sample_splits = [2]

    sample_leafs = None
    if min_samples_leaf is not None:
        sample_leafs = min_samples_leaf
    else:
        sample_leafs = [1]

    for crit in crits:
        DTModel= tree.DecisionTreeClassifier( criterion=crit )
        print( "Tuning the criterion. Current criterion being tested:", crit,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        DTModel.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = DTModel.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[0].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for criterion achieved!" )
            best_criterion = crit
        print()

    print( "Finished tuning criterion. Moving on to tuning the splitter." )
    print( "-"*40 )
    best_acc = 0
 
    for split in splits:
        DTModel = tree.DecisionTreeClassifier( splitter=split )
        print( "Tuning the splitter. Current splitter being tested:", split,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        DTModel.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = DTModel.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[1].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for splitter achieved!" )
            best_splitter = split
        print() 
    
    print( "Finished tuning the splitter. Moving on to tuning the min split size." )
    print( "-"*40 )
    best_acc = 0

    for ssplit in sample_splits:
        DTModel = tree.DecisionTreeClassifier( min_samples_split=ssplit )
        print( "Tuning minimum samples split. Current min split being tested:", ssplit,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        DTModel.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = DTModel.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[2].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for min split achieved!" )
            best_samples_split = ssplit
        print()
    
    print( "Finished tuning the min split. Moving on to tuning the min leaf size." )
    print( "-"*40 )
    best_acc = 0
    
    for leaf in sample_leafs:
        DTModel= tree.DecisionTreeClassifier( min_samples_leaf=leaf )
        print( "Tuning min samples leaf size. Current leaf size being tested:", leaf,
                ". Time stamp:" , str( datetime.datetime.now() ) )
        DTModel.fit( X_train, y_train )
        print( "Finished fitting. Starting scoring on validation set. Time stamp:", str( datetime.datetime.now() ) )
        acc = DTModel.score( X_val, y_val )
        print( "Finished scoring. Here is the accuracy:", str( acc*100 ) + "%", "Time stamp:", str( datetime.datetime.now() ) )
        acc_train[3].append( acc )
        if acc > best_acc:
            best_acc = acc
            print( "Current best accuracy for leaf size achieved!" )
            best_samples_leaf = leaf
        print()

    print( "Finished tuning the min leaf size. Building the overall best model." )
    print( "-"*40 )

    model = tree.DecisionTreeClassifier( criterion=best_criterion, splitter=best_splitter,
            min_samples_split=best_samples_split, min_samples_leaf=best_samples_leaf )
    model.fit( X_train, y_train )
    print( "Finished building the model, starting scoring against the validation set. Time stamp:", str( datetime.datetime.now() ) )
    
    model_acc = model.score( X_val, y_val )
    print( "Finished building and validation the model. Time stamp:", str( datetime.datetime.now() ) )
    return model, model_acc, acc_train, best_criterion, best_splitter, best_samples_split, best_samples_leaf


def setUpData(file_path):

    # Read in the data. The dataset I am working with is actually tab separated
    df = pd.read_csv( file_path, delimiter='\t', nrows=None )
    # Now, I only need the first 50 columns as that is the feature set I want
    # to work with. 
    #df = df.iloc[:, range( 50 ) ]
    df = df.iloc[:, range( 50 ) ]
    # Clean up the dataset of rows with NaN's as that causes the conversion to 
    # int for labels to become wrong.
    df = df.dropna()
    
    # I prefer to work with numpy arrays as I have not worked with Pandas before.
    data = df.to_numpy()
    # Drop all the meaningless data.
    data = data[~np.any( data == 0, axis=1 ) ]

    # Shuffling the data consistently with "pi" as the seed because I like the 
    # number pi.
    np.random.seed( 314159 )
    np.random.shuffle( data )
    
    # Since I decided to predict the last column values from the rest of the 
    # dataset, I am splitting X and y as given below. Convert y values to int
    # as the labels are not supposed to be floats.
    y = data[:, 16 ]
    X = data[:, 10:20]
    X = np.delete( X, 6, 1)
    y = y.astype( int )

    # Print the shapes of the original data

    # Split the data into the train - validate - test sets to do hyperparameter
    # tuning properly. I rescale the X values to be slightly more meaningful 
    # for the learning algorithms. This is following the suggestion from the
    # SciKit Learn documentation.
    X_train, y_train = X[:200000]/5, y[:200000] 
    X_val, y_val = X[200000:210000]/5, y[200000:210000]
    X_test, y_test = X[210000:220000]/5, y[210000:220000]
    # Print the shapes as that is useful to just keep in mind when working with
    # the data. 
    print( "Train data shapes:", X_train.shape, y_train.shape )
    print( "Validation data shapes:", X_val.shape, y_val.shape )
    print( "Test data shapes:", X_test.shape, y_test.shape )
    return X_train, y_train.ravel(), X_val, y_val.ravel(), X_test, y_test.ravel()


def main():
    if len( sys.argv ) < 2:
        print( "How to use the script:" )
        print( "\tpython3 big_five_classification_approaches.py path/to/dataset [tune]" )
        print( "\tThe [tune] parameter is optional, and will indicate that you want",
                "to retune all the hyper parameters. This can take a long time" )
        sys.exit()

    X_train, y_train, X_val, y_val, X_test, y_test =  \
            setUpData( str( sys.argv[1] ) ) 
    
    # Previously tuned hyper parameters. Used if the user does not want to 
    # retune the hyper parameters. 
    KNN_best_n_neighbors, KNN_best_leaf_size, KNN_best_p = 50, 25, 2
    NN_best_activation, NN_best_solver, NN_best_alpha, NN_best_dim = "relu", "adam", 0.01, (200,)
    DTModel_best_criterion, DTModel_best_splitter, DTModel_best_samples_split, DTModel_best_samples_leaf = "gini", "best", 194, 131 
    KNNModel = None
    NNModel = None
    DTModel = None
    KNNModel_val_acc = 0
    NNmodel_val_acc = 0
    DTModel_val_acc = 0
    KNNModel_test_acc = 0 
    
    if len( sys.argv ) < 3:
        KNNModel = neighbors.KNeighborsClassifier( n_neighbors=KNN_best_n_neighbors,
                leaf_size=KNN_best_leaf_size, p=KNN_best_p )
        print( "Building the model with the tuned hyper parameters for KNN" )
        KNNModel.fit( X_train, y_train )
        print( "Done training, calculating prediction accuracies." )
        KNNModel_val_acc = KNNModel.score( X_val, y_val )
        KNNModel_test_acc = KNNModel.score( X_test, y_test )

        NNModel = neural_network.MLPClassifier( max_iter=100, hidden_layer_sizes=NN_best_dim,
                alpha=NN_best_alpha, solver=NN_best_solver, activation=NN_best_activation )
        print( "Building the model with the tuned hyper parameters for NN" )
        NNModel.fit( X_train, y_train )
        print( "Done training, calculating prediction accuracies." )
        NNmodel_val_acc = NNModel.score( X_val, y_val )

        DTModel = tree.DecisionTreeClassifier( criterion=DTModel_best_criterion,
                splitter=DTModel_best_splitter, min_samples_split=DTModel_best_samples_split,
                min_samples_leaf=DTModel_best_samples_leaf )
        print( "Building the model with the tuned hyper parameters for DT" )
        DTModel.fit( X_train, y_train )
        print( "Done training, calculating prediction accuracies." )
        DTModel_val_acc = DTModel.score( X_val, y_val )

    else:
        # You can modify the hyper parameter list to tune from here. These are
        # the hyper parameters for the KNN classifier.
        n = [1] + list( range( 2, 52, 2 ) )
        l = [1] + list( range( 2, 36 ) )
        p = [1, 2]
        KNNModel, KNNModel_val_acc, KNN_acc_train, KNN_best_n_neighbors, KNN_best_leaf_size, KNN_best_p = KNNClassifierWithTuning( X_train, y_train, X_val, y_val, n, l, p )
        KNNModel_test_acc = KNNModel.score( X_test, y_test )

        # Save all the training accuracy plots for KNN
        plt.plot( n, KNN_acc_train[0] )
        plt.xlabel( "Number of Neighbors" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "n_neigh_knn_acc_train.jpg" )
        plt.clf()
        plt.plot( l, KNN_acc_train[1] )
        plt.xlabel( "Leaf Size" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "leaf_size_knn_acc_train.jpg" )
        plt.clf()
        plt.plot( p, KNN_acc_train[2] )
        plt.xlabel( "Lp Norm Regularization" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "p_knn_acc_train.jpg" )
        
        # You can modify the hyper parameter list to tune from here. These are
        # the hyper parameters for the MLP Classifier (Neural Network)
        activation = [ "identity", "logistic", "tanh", "relu" ]
        solver = [ "lbfgs", "sgd", "adam" ]
        alpha = [ 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3 ]
        dims = [ (25,), (50,), (75,), (100,), (125,), (150,), (175,), (200,), (225,) ]
        dims_draw = [25, 50, 75, 100, 125, 150, 175, 200, 225]

        NNModel, NNmodel_val_acc, NN_acc_train, NN_best_activation, NN_best_solver, NN_best_alpha, NN_best_dim = NNClassifierWithTuning( X_train, y_train, X_val, y_val, activation, solver, alpha, dims ) 

        # Save all the training accuracy plots for the MLP Classifier.
        plt.clf()
        plt.plot( activation, NN_acc_train[0] )
        plt.xlabel( "Activation Function" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "activation_nn_acc_train.jpg" )
        plt.clf()
        plt.plot( solver, NN_acc_train[1] )
        plt.xlabel( "Solver" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "solver_nn_acc_train.jpg" )
        plt.clf()
        plt.plot( alpha, NN_acc_train[2] )
        plt.xlabel( "Regularization constant" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "alpha_nn_acc_train.jpg" )
        plt.clf()
        plt.plot( dims_draw, NN_acc_train[3] )
        plt.xlabel( "NN Hidden Layer Size" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "dim_nn_acc_train.jpg" )
        plt.clf()
        
        # You can modify the hyper parameter list to tune from here. These are
        # the hyper parameters for the Decision Tree Classifier.
        criterion = ["gini", "entropy"]
        splitter = ["best", "random"]
        min_samples_split= list( range( 2, 200, 2 ) )
        min_samples_leaf = list( range( 1, 200, 2 ) )

        DTModel, DTModel_val_acc, DTModel_acc_train, DTModel_best_criterion, DTModel_best_splitter, DTModel_best_samples_split, DTModel_best_samples_leaf = DTreeClassifierWithTuning( 
                X_train, y_train, X_val, y_val, criterion, splitter, min_samples_split, min_samples_leaf )
        
        # Save all the training accuracy plots for the DT Classifier.
        plt.clf()
        plt.plot( criterion, DTModel_acc_train[0] )
        plt.xlabel( "Criterion" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "dt_criterion.jpg" )
        plt.clf()
        plt.plot( splitter, DTModel_acc_train[1] )
        plt.xlabel( "Splitter" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "dt_splitter.jpg" )
        plt.clf()
        plt.plot( min_samples_split, DTModel_acc_train[2] )
        plt.xlabel( "Minimum Samples Split Size" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "dt_split_size.jpg" )
        plt.clf()
        plt.plot( min_samples_leaf, DTModel_acc_train[3] )
        plt.xlabel( "Minimum Samples Leaf Size" )
        plt.ylabel( "Accuracy" )
        plt.savefig( "dt_leaf_size.jpg" )
        plt.clf()
        

    # Print out all the results. 
    print( "K-Nearest-Neighbors Results:" )
    print( "\tKNN Model Validation Accuracy:", str( KNNModel_val_acc * 100 ) + "%" )
    print( "\tKNN Model Test Accuracy: ", str( KNNModel_test_acc * 100 ) + "%" )
    print( "\tKNN Model Best Hyper Parameters:" )
    print( "\t\tNumber of Neighbors:", KNN_best_n_neighbors )
    print( "\t\tLeaf Size:", KNN_best_leaf_size )
    print( "\t\tBest L-p Norm:", KNN_best_p )
    print("-"*40)
    print()
    print( "Neural Network Results:" )
    print( "\tNN Model Validation Accuracy:", str( NNmodel_val_acc*100 ) + "%" )
    print( "\tNN Model Test Accuracy:", str( NNModel.score( X_test, y_test ) * 100 ) + "%" )
    print( "\tNN Model Best Hyper Parameters:" )
    print( "\t\tActivation Function:", NN_best_activation )
    print( "\t\tSolver Function:", NN_best_solver )
    print( "\t\tRegularization Alpha:", NN_best_alpha )
    print( "\t\tHidden Layer Size:", NN_best_dim )
    print( "-"*40 )
    print()
    print( "Decision Tree Results:" )
    print( "\tDTModel validation accuracy:", str( DTModel_val_acc*100 ) + "%" )
    print( "\tDTModel test perf:", str( DTModel.score( X_test, y_test ) * 100 ) + "%" )
    print( "\tDTModel best hyper parameters:" )
    print( "\t\tCriterion:", DTModel_best_criterion )
    print( "\t\tSplitter:", DTModel_best_splitter ) 
    print( "\t\tSamples Split:", DTModel_best_samples_split ) 
    print( "\t\tSamples Leaf:", DTModel_best_samples_leaf )
    print( "-"*40 )
    print()
    print( "Trivial Baseline Results (Majority Guess)" )
    
    freq = dict()
    for i in y_train:
        if i not in freq:
            freq[i] = 0
        freq[i] += 1

    for i in freq:
        print( "\tLabel", i, "Occurs" , str((freq[i]/len(y_train))*100) + "%", "of the time in the training data." )
    

main()
