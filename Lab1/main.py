import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from lab_ML_supervised import \
    rand_gauss, rand_bi_gauss, rand_clown, rand_checkers, grid_2d, plot_2d, \
    frontiere, mse_loss, gradient, plot_gradient, poly2, collist, \
    symlist, gr_mse_loss, hinge_loss, gr_hinge_loss

from Data_generation import *
from Logistic_regression import *

########################################## marianne.clausel@imag.fr
### Main script

def main() :
        #plot_random(100)
        dataX,dataY = save_random(100,100)
        #plot_2d(dataX,dataY)
        #plt.show()
        for i in range(0,3):
            print(i,"- size dataX = ",dataX[0].size," ; size dataY = ",dataY[0].size,"\n")
            logisticRegression(dataX[i],dataY[i])
        return 0



if __name__ == "__main__":
	main()
