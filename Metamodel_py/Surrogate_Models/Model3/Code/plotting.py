# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:24:26 2022

@author: yairn
"""

# import numpy as np
import matplotlib.pyplot as plt

import definitions

plots = definitions.plots
#################################################
# Set what data to plot:


def plotData(DataToPlot, plotWhat, submodelName):

    # titles and labels:
    xLabel = plots['xLabel']
    yLabel = plots['yLabel']

    # Titles for the subplots:
    colTitles = plots[submodelName]['title']
    rowTitles = plots['rowTitles']

    # min and max values for the different heatmaps:
    vmins = plots[submodelName]['vmin']
    vmaxs = plots[submodelName]['vmax']
    contourLevels = plots[submodelName]['contourLevels']

    # Number of rows and columns of subplots:
    nRows = plots['nRows']
    nCols = plots['nCols']

    # Plot a row of subplot if the data is not empty and if value is 'True':
    for iRow in range(nRows):
        if DataToPlot[iRow] is not None and plotWhat[iRow]:
            plotHeatmaps(data=DataToPlot[iRow],
                         nRows=nRows,
                         nCols=nCols,
                         rowTitles=rowTitles,
                         colTitles=colTitles,
                         xLabel=xLabel,
                         yLabel=yLabel,
                         vmins=vmins,
                         vmaxs=vmaxs,
                         contourLevels=contourLevels,
                         iRow=iRow)
#################################################
# Plot heatmaps subplots


def plotHeatmaps(
        data,
        rowTitles,
        colTitles,
        xLabel,
        yLabel,
        nRows,
        nCols,
        vmins,
        vmaxs,
        contourLevels,
        iRow):

    fig = plt.figure(figsize=plots['figSize'])

    # im identifier:
    im = [None]*nCols
    x1, x2 = data[0]  # Free parameters of the data.
    f = data[1]  # Data that is the function of the free parameters.

    # Return the last row that is 'True' in 'plotWhat'. It sets in what row
    # to show 'xlabel':

    max_plotWhat = 4  # np.max(np.where(plotWhat))

    colormap = plots['colormap']
    fontsize1 = plots['fontSizes1']

    # plot the nRows x nCols subplots with labels, titles at
    # sceciefic locations. iCol is Column index, iRow is Row index:
    for iCol in range(nCols):
        fig.add_subplot(nRows, nCols, iRow*nCols + iCol+1)
        im[iCol] = plt.pcolor(x1, x2, f[iCol],
                              vmin=vmins[iCol],
                              vmax=vmaxs[iCol],
                              shading='auto',
                              cmap=colormap)
        if True:  # iRow > 0:
            cs = plt.contour(x1, x2, f[iCol],
                             contourLevels,
                             colors='k',
                             vmin=vmins[iCol],
                             vmax=vmaxs[iCol])
            plt.clabel(cs, contourLevels, inline=True, fmt='%.1f',
                       fontsize=fontsize1)

        fig.colorbar(im[iCol])

        if iRow == 0:
            plt.title(colTitles[iCol] + rowTitles[iRow])

        else:
            plt.title(rowTitles[iRow])

        if iRow == max_plotWhat:
            plt.xlabel(xLabel)

        # plt.axis('equal')
#################################################
