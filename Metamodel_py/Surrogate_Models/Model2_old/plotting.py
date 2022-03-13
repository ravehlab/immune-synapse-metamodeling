# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:24:26 2022

@author: yairn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

#################################################
# Set what data to plot:


def plotData(DataToPlot, plotWhat):
    # titles and labels:
    xLabel = "$Poff_{LCK*}^{LCKA}$"
    yLabel = "$D_{LCK*}^{LCKA} [\mu m^2/sec]$"
    DLALck_Title = "$Decaylength_{Lck*}^{LckA}[nm]$\n"  # Decaylength
    colTitles = [DLALck_Title]

    rowTitles = ["Training data", "Fit data",
                 "Trained parameters", "Surrogate"]

    # min and max values for the different heatmaps:
    vmins = [0]
    vmaxs = [3.0]

    nRows = 4
    nCols = 1

    # Plot a row of subplot if the data is not empty and if value is 'True':
    for iRow in range(nRows):
        if DataToPlot[iRow]!=None and plotWhat[iRow]:
            plotHeatmaps(data=DataToPlot[iRow],
                         nRows=nRows,
                         nCols=nCols,
                         rowTitles=rowTitles,
                         colTitles=colTitles,
                         xLabel=xLabel,
                         yLabel=yLabel,
                         vmins=vmins,
                         vmaxs=vmaxs,
                         iRow=iRow)
#################################################
# Plot heatmaps subplots


def plotHeatmaps(data,
                 rowTitles,
                 colTitles,
                 xLabel,
                 yLabel,
                 nRows,
                 nCols,
                 vmins,
                 vmaxs,
                 iRow):

    ticklablesFontsize = 14
    cbarFontSize = 14

    fig = plt.figure(figsize=[4, 12])

    # im identifier:
    im = [None]*nCols
    x1, x2 = data[0]  # Free parameters of the data.
    f = data[1]  # Data that is the function of the free parameters.

    # Return the last row that is 'True' in 'plotWhat'. It sets in what row
    # to show 'xlabel':
    max_plotWhat = 4

    # plot the nRows x nCols subplots with labels, titles at
    # sceciefic locations. iCol is Column index, iRow is Row index:
    for iCol in range(nCols):
        ax = fig.add_subplot(nRows, nCols, iRow*nCols + iCol+1)
        im[iCol] = plt.pcolor(10**x1, 10**x2, 10**f[iCol],
                              shading='auto', cmap='Blues',
                              norm=colors.LogNorm())

        if True:  # iRow > 0:
            DL_levels = np.arange(0., 3., 0.25)
            cs = plt.contour(10**x1, 10**x2, 10**f[iCol],
                             10**DL_levels, colors='k',
                             vmin=vmins[iCol], vmax=vmaxs[iCol])
            plt.clabel(cs, 10**DL_levels, inline=True, fmt='%.0f', fontsize=10)

        cbar0 = fig.colorbar(im[iCol])
        cbar0.ax.set_title('nm', fontsize=cbarFontSize)

        ax.set_xlim(10**-5.2, 10**-0.4)
        ax.set_xticklabels(x1, fontsize=ticklablesFontsize)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))

        ax.set_ylim(10**-3.1, 10**-0.7)
        ax.set_yticklabels(x2, fontsize=ticklablesFontsize)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=7))

        if iRow == 0:
            plt.title(colTitles[iCol] + rowTitles[iRow])

        else:
            plt.title(rowTitles[iRow])

        if iRow == max_plotWhat:
            plt.xlabel(xLabel)

        if iCol == 0:
            plt.ylabel(yLabel)

    fig.tight_layout()
#################################################

    