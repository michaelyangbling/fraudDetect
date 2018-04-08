#to plot corr: plotCorr.plot
import seaborn as sns
import matplotlib.pyplot as plt
cor=None
def plot(data):
    global cor
    cor = data.corr()
    sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, linewidths=2, cmap="PiYG")
#to reload
#import imp
#imp.reload(plotCorr)
