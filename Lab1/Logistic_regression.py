import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from sklearn.linear_model import *
from sklearn import linear_model

#Q3 -
""" Attribut de LogisticRegression()
    coef_ : array, shape (n_classes, n_features)

    Coefficient of the features in the decision function.

intercept_ : array, shape (n_classes,)

    Intercept (a.k.a. bias) added to the decision function. If fit_intercept is set to False, the intercept is set to zero.

n_iter_ : array, shape (n_classes,) or (1, )

    Actual number of iterations for all classes. If binary or multinomial, it returns only 1 element. For liblinear solver, only the maximum number of iteration across all classes is given.
   """
#Q4 -
"""   predict(X) =  	Predict class labels for samples in X.  """
"""   score(X, y[, sample_weight]) =   Returns the mean accuracy on the given test data and labels.     """


def logisticRegression(dataX,dataY) :
        h = .02  # step size in the mesh

        my_log = linear_model.LogisticRegression()
        d = my_log.fit(dataX,dataY)
        print(" \n ",d)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = dataX[:, 0].min() - .5, dataX[:, 0].max() + .5
        y_min, y_max = dataX[:, 1].min() - .5, dataX[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = my_log.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(dataX[:, 0], dataX[:, 1], c=dataY, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.show()
