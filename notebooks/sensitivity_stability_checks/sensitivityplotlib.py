import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

def plot_cm_effect_sensitivity(filenames, 
                               cm_labels,
                               legend_labels, 
                               figsize=(5,3.3), 
                               colors = sns.color_palette("colorblind"),
                               y_offset=0.08, 
                               y_scale=1.3,
                               fontsize=8,
                               xlims = [-100,100]):
    
    fig = plt.figure(figsize=figsize, dpi=300)
    for i in range(len(filenames)):
        # load trace
        cm_trace = np.loadtxt(filenames[i])
    
        # combine traces for overlapping features
        cm_trace[:,4] = cm_trace[:,3]*cm_trace[:,4]
        cm_trace[:,5] = cm_trace[:,4]*cm_trace[:,5]
        cm_trace[:,7] = cm_trace[:,6]*cm_trace[:,7]
        
        # calculate means and confidence intervals
        means = 100*(1 - np.mean(cm_trace, axis=0))
        li = 100 * (1 - np.percentile(cm_trace, 2.5, axis=0))
        ui = 100 * (1 - np.percentile(cm_trace, 97.5, axis=0))
        lq = 100 * (1 - np.percentile(cm_trace, 25, axis=0))
        uq = 100 * (1- np.percentile(cm_trace, 75, axis=0))
        N_cms = means.size
        
        y_vals = -1 * y_scale * np.arange(N_cms)
        plt.plot(means, y_vals+ i*y_offset, marker="|", markersize=10, color=colors[i], label = legend_labels[i],
                 linewidth=0)
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], [y_vals[cm]+ i*y_offset, y_vals[cm]+ i*y_offset], color=colors[i], alpha=0.25)
            plt.plot([lq[cm], uq[cm]], [y_vals[cm]+ i*y_offset, y_vals[cm]+ i*y_offset], color=colors[i], alpha=0.5)
            
    plt.plot([0, 0], [1, -(N_cms)*y_scale], "--r", linewidth=0.5)
    xtick_vals = np.arange(-100, 150, 50)
    xtick_str = [f"{x:.0f}%" for x in xtick_vals]
    plt.yticks(y_vals, np.array(cm_labels))
    plt.xticks(xtick_vals, xtick_str)
    plt.xlim(xlims)
    plt.ylim([y_vals[-1]-y_scale/2, y_vals[0]+y_scale*.6])
    plt.xlabel("Percentage Reduction in $R$", fontsize=fontsize)
    plt.legend(frameon=False, fontsize=fontsize, loc = 'upper right')
    plt.rc('font', size=fontsize)
    sns.despine()
    plt.tight_layout()   