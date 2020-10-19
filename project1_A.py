#######################################################################################################################

####################################
# Funny Money A: Database analysis #
# Created by Aniket N Prabhu       #
# References: ML_datanalysis2.py   #
####################################

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data
import matplotlib.pyplot as plt         # needed for plotting
from matplotlib import cm as cm         # needed for the color map
import seaborn as sns                   # needed data visualization (pairs plot)

# Defining a function for correlation #################################################################################
# Dataframe is a dictionary-like container and is a pandas feature


def highcorr(datfram, repnum):
    cormat = datfram.corr()  # correlation matrix
    print(cormat)

    # ".corr()" computes pairwise correlation of columns, excluding any null/NA values. This finds the correlations

    cormat *= np.tri(*cormat.values.shape, k=-1).T

    # "x*=y" corresponds to "x=x*y". "np.tri" creates lower triangular matrix with 1s on &/or below the diagonal(diag).
    # It also unpacks the tuple into rows and columns. k=-1 means that the diag of 0s and 1s below the diag.

    # print(cormat)  # For debugging
    cormat = cormat.stack()  # Reorganizes columns into rows
    # print(cormat)  # For debugging

    cormat = cormat.reindex(cormat.abs().sort_values(ascending=False).index).reset_index()

    # ".abs()" returns absolute numeric values of each element in the dataframe.
    # ".sort_values" will sort values in descending order since "ascending=False".
    # ".index" returns the index/row labels of the array.
    # ".reset_index()" resets the indices.
    # ".reindex()" conform DataFrame to new index with optional filling logic.
    # print(cormat)  # For debugging

    # Assigning column names.
    cormat.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    print("\nMost Highly Correlated")
    print(cormat.head(repnum))  # prints the top values


#######################################################################################################################

# Heat map ############################################################################################################


def heatmap(df):  # Although not explicitly asked, I don't intend to make compromises.
    # creating a figure that's 7x7 units with 100 dots per inch
    fig = plt.figure(figsize=(7, 7), dpi=100)

    # add a subplot that has 1 row, 1 column, and is the first subplot
    ax1 = fig.add_subplot(111)

    # get the 'jet' color map
    cmap = cm.get_cmap('jet', 30)

    # Perform the correlation and take the absolute value of it. Then map
    # the values to the color map using the "nearest" value
    cax = ax1.imshow(np.abs(df.corr()), interpolation='nearest', cmap=cmap)

    # now set up the axes
    major_ticks = np.arange(0, len(df.columns), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True, which='both', axis='both')
    plt.title('Correlation Matrix')
    ax1.set_xticklabels(df.columns, fontsize=9)
    ax1.set_yticklabels(df.columns, fontsize=12)

    # add the legend and show the plot
    fig.colorbar(cax, ticks=[-0.4, -0.25, -.1, 0, 0.1, .25, .5, .75, 1])
    plt.show()

#######################################################################################################################

# Function to create pairs plot #######################################################################################


def prplt(dfram):
    sns.set(style='whitegrid', context='notebook')  # Setting appearance.
    # color = ['amber', 'violet']
    sns.pairplot(dfram, hue='Class', height=2.5)    # creates pairs plot.
    plt.show()                                      # Shows the plot.


# Using above defined functions to achieve our goals #

funmon = pd.read_csv('data_banknote_authentication.txt', names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class'])
funmon.to_csv('data_banknote_authentication.csv')   # converts to ".csv" format so that we may add hue in the pairs plot
highcorr(funmon, 10)  # No. of ways of choosing 2 objects from 5 objects is 5C2=10
heatmap(funmon)
prplt(funmon)

#######################################################################################################################
# End #################################################################################################################
#######################################################################################################################
