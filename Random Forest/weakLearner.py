

#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes.
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats
from numpy import inf
import random

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:


        """
        #print "   "
        pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#
        return score, Xlidx,Xridx

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#	
	

        #---------End of Your Code-------------------------#

    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        
        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...
        
        # YOUR CODE HERE
        

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using
        a random set of features from the given set of features...

    """

    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
	self.fidx=-1
	self.split=-1
        #pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        #print "Inside the train of Random"
        nexamples,nfeatures=X.shape

        #print "Train has X of length ", X.shape


        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
	


        #---------End of Your Code-------------------------#
    

    def findBestRandomSplit(self,feat,Y):
        """

            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)


        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
	
        #---------End of Your Code-------------------------#
        return splitvalue, minscore, Xlidx, Xridx

    def calculateEntropy(self, Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

       
        return sentropy



    def evaluate(self, X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
	#print X.shape, self.fidx, "xshape"
	

        #---------End of Your Code-------------------------#




# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
	self.a=0
	self.b=0
	self.c=0
	self.F1=0
	self.F2=0
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...

        """
        RandomWeakLearner.__init__(self,nsplits)

        #pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible

            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
	
	    
        #---------End of Your Code-------------------------#
        return 0, minscore, bXl, bXr


    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
	

        #---------End of Your Code-------------------------#


#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
	self.a=0
	self.b=0
	self.c=0
	self.d=0
	self.e=0	
	self.f=0
	self.F1=0
	self.F2=0
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        a, b, c, d, e, f = np.random.uniform(-3, 3, (6,))
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
	
        #---------End of Your Code-------------------------#
        return 0, minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
	
        #---------End of Your Code-------------------------#
