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
                               xlims = [-100,100],
                               legend_title=None,
                               legend_fontsize=8,
                               leavouts=False,
                               bbox_to_anchor=None,
                               combine_hierarchical=True):
    
    fig = plt.figure(figsize=figsize, dpi=300)
    
    
    ind_1000 = np.where(np.array(cm_labels)=='Gatherings <1000')[0][0]
    ind_100 = np.where(np.array(cm_labels)=='Gatherings <100')[0][0]
    ind_10 = np.where(np.array(cm_labels)=='Gatherings <10')[0][0]
    ind_some_bus = np.where(np.array(cm_labels)=='Some Businesses Suspended')[0][0]
    ind_most_bus = np.where(np.array(cm_labels)=='Most Businesses Suspended')[0][0]    
    
    for i in range(len(filenames)):
        # load trace
        cm_trace = np.loadtxt(filenames[i])
        
        # combine traces for overlapping features
        if combine_hierarchical==True:
            # if plotting leavouts, have to combine differently
            if leavouts==True:
                if (legend_labels[i]=='Healthcare Infection Control' or 
                    legend_labels[i]=='Mask Wearing' or
                    legend_labels[i]=='Symptomatic Testing'):
                    # everything below is shifted up
                    cm_trace[:,ind_100-1] = cm_trace[:,ind_1000-1]*cm_trace[:,ind_100-1]
                    cm_trace[:,ind_10-1] = cm_trace[:,ind_100-1]*cm_trace[:,ind_10-1]
                    cm_trace[:,ind_most_bus-1] = cm_trace[:,ind_some_bus-1]*cm_trace[:,ind_most_bus-1]
                elif legend_labels[i]=='Gatherings <1000':
                    # shift up, while removing the combos with gatherings<1000
                    cm_trace[:,ind_100-1] = cm_trace[:,ind_100-1]
                    cm_trace[:,ind_10-1] = cm_trace[:,ind_100-1]*cm_trace[:,ind_10-1]
                    cm_trace[:,ind_most_bus-1] = cm_trace[:,ind_some_bus-1]*cm_trace[:,ind_most_bus-1]
                elif legend_labels[i]=='Gatherings <100':
                    # gatherings<100 doesn't exist
                    # shift up, while removing the combos with gatherings<100
                    # note that gatherings<1000 will not be shifted up
                    cm_trace[:,ind_10-1] = cm_trace[:,ind_1000]*cm_trace[:,ind_10-1]
                    cm_trace[:,ind_most_bus-1] = cm_trace[:,ind_some_bus-1]*cm_trace[:,ind_most_bus-1]
                elif legend_labels[i]=='Gatherings <10':
                    # gatherings<100 no longer needs to be shifted up
                    # gatherings<10 doesn't exist
                    # most businesses shifted up
                    cm_trace[:,ind_100] = cm_trace[:,ind_1000]*cm_trace[:,ind_100]
                    cm_trace[:,ind_most_bus-1] = cm_trace[:,ind_some_bus-1]*cm_trace[:,ind_most_bus-1]
                elif legend_labels[i]=='Some Businesses Suspended':
                    # gatherings<100 no longer needs to be shifted up
                    # gatherings<10 no longer needs to be shifted up
                    # some businesses doesn't exist, so shift most bus up
                    cm_trace[:,ind_100] = cm_trace[:,ind_1000]*cm_trace[:,ind_100]
                    cm_trace[:,ind_10] = cm_trace[:,ind_100]*cm_trace[:,ind_10]
                    cm_trace[:,ind_most_bus-1] = cm_trace[:,ind_most_bus-1]
                elif legend_labels[i]=='Most Businesses Suspended':
                    # gatherings don't need to be shifted
                    # most businesses doesn't exist
                    cm_trace[:,ind_100] = cm_trace[:,ind_1000]*cm_trace[:,ind_100]
                    cm_trace[:,ind_10] = cm_trace[:,ind_100]*cm_trace[:,ind_10]
                else:
                    # if leavout is None, School closure, or stay at home order the combos are normal

                    cm_trace[:,ind_100] = cm_trace[:,ind_1000]*cm_trace[:,ind_100]
                    cm_trace[:,ind_10] = cm_trace[:,ind_100]*cm_trace[:,ind_10]
                    cm_trace[:,ind_most_bus] = cm_trace[:,ind_some_bus]*cm_trace[:,ind_most_bus]    
            else:
                cm_trace[:,ind_100] = cm_trace[:,ind_1000]*cm_trace[:,ind_100]
                cm_trace[:,ind_10] = cm_trace[:,ind_100]*cm_trace[:,ind_10]
                cm_trace[:,ind_most_bus] = cm_trace[:,ind_some_bus]*cm_trace[:,ind_most_bus]
        
        # calculate means and confidence intervals
        means = 100*(1 - np.mean(cm_trace, axis=0))
        li = 100 * (1 - np.percentile(cm_trace, 2.5, axis=0))
        ui = 100 * (1 - np.percentile(cm_trace, 97.5, axis=0))
        lq = 100 * (1 - np.percentile(cm_trace, 25, axis=0))
        uq = 100 * (1- np.percentile(cm_trace, 75, axis=0))
        N_cms = len(cm_labels)
        
        # if plotting leavouts, add nan to indices of leftout CMs
        if leavouts==True:
            if legend_labels[i]=='None':
                pass
            else:
                ind = np.where(np.array(cm_labels)==legend_labels[i])[0][0]
                means = np.insert(means, ind, np.nan)
                li = np.insert(li, ind, np.nan)
                ui = np.insert(ui, ind, np.nan)
                lq = np.insert(lq, ind, np.nan)
                uq = np.insert(uq, ind, np.nan)
        # if not plotting a leavout set, but extra CMs added to end (e.g mobility)
        if leavouts==False and len(means)<len(cm_labels):
            for k in range(len(cm_labels)-len(means)): # assumes extra CMs are at the end
                means = np.append(means,np.nan)
                li = np.append(li,np.nan)
                ui = np.append(ui,np.nan)
                lq = np.append(lq,np.nan)
                uq = np.append(uq,np.nan)
            
        
        # plot shading to make easier to see
        y_vals = -1 * y_scale * np.arange(N_cms)
        xrange = np.array(xlims)
        for j in range(0, N_cms, 2):
            plt.fill_between(xrange, y_vals[j]+y_scale/2, y_vals[j]-y_scale/2, color="whitesmoke")        
        
        # plot data
        shift_center = y_offset*len(legend_labels)/2
        height= len(filenames)-i
        plt.plot(means, y_vals+ height*y_offset - shift_center, marker='|', markersize=10, color=colors[i], label = legend_labels[i],
                 linewidth=0)
        for cm in range(N_cms):
            plt.plot([li[cm], ui[cm]], 
                     [y_vals[cm]+ height*y_offset-shift_center, y_vals[cm]+ height*y_offset-shift_center],
                     color=colors[i], alpha=0.25)
            plt.plot([lq[cm], uq[cm]], 
                     [y_vals[cm]+ height*y_offset-shift_center, y_vals[cm]+ height*y_offset-shift_center], 
                    color=colors[i], alpha=0.5)
            
    plt.plot([0, 0], [1, -(N_cms)*y_scale], "--k", linewidth=0.5)
    xtick_vals = np.arange(-100, 150, 50)
    xtick_str = [f"{x:.0f}%" for x in xtick_vals]
    plt.yticks(y_vals, np.array(cm_labels))
    plt.xticks(xtick_vals, xtick_str)
    plt.xlim(xlims)
    plt.ylim([y_vals[-1]-y_scale/2, y_vals[0]+y_scale/2])
    plt.xlabel("Percentage Reduction in $R$", fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize, loc = 'upper right', 
               title=legend_title, title_fontsize=legend_fontsize, bbox_to_anchor=bbox_to_anchor,
               shadow=True, fancybox=True)
    plt.rc('font', size=fontsize)
    sns.despine()
    plt.tight_layout()   