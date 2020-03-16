import matplotlib.pyplot as plt
import numpy as np
import mlai
import IPython
import os
#from IPython.display import display, clear_output, HTML

tau = 2*np.pi

two_figsize = (10, 5)
one_figsize = (5, 5)
big_figsize = (7, 7)
wide_figsize = (7, 3.5)
hcolor = [1., 0., 1.] # highlighting color

# def write_plots(filename=filename, filebase, directory=None, width=700, height=500, kwargs):
#     """Display a series of plots controlled by sliders. The function relies on Python string format functionality to index through a series of plots."""
#     args = collections.OrderedDict(kwargs)
    
#     def write_figure(filebase, directory, **kwargs):
#         """Helper function to load in the relevant plot for display."""
#         filename = filebase.format(**kwargs)
#         if directory is not None:
#             filename = directory + '/' + filename
#         return "<img src='{filename}'>".format(filename=filename)
#     meta = '{data-transition="None"}'
#     out = '### ' + meta
#     for name, val in kwargs.items():
#         if isinstance(val,list) or isinstance(
#         out += '\n\n' + write_figure(filebase=filebase, directory=directory, **kwargs) + '\n\n'
#         for 
#     interact(show_figure, filebase=fixed(filebase), directory=fixed(directory), **kwargs)


def matrix(A, ax=None,
                bracket_width=3,
                bracket_style='square',
                type='values',
                colormap=None,
                highlight=False,
                highlight_row=None,
                highlight_col=None,
                highlight_width=3,
                highlight_color=[0,0,0],
                zoom=False,
                zoom_row=None,
                zoom_col=None,
                bracket_color=[0,0,0],
                fontsize=16):
    """Plot a matrix for visualisation in a slide or piece of text."""
    
    if ax is None:
        ax = plt.gca()

    if colormap is not None:
        plt.set_cmap(colormap) 

    A = np.asarray(A)
    
    nrows = A.shape[0]
    ncols = A.shape[1]
    
  
    x_lim = np.array([-0.75, ncols-0.25])
    y_lim = np.array([-0.75, nrows-0.25])
  
    ax.cla()
    handle=[]
    if type == 'image':
        handle =  ax.matshow(A)
    elif type == 'imagesc':
        handle =  ax.images(A, [np.array([A.min(), 0]).min(), A.max()])
    elif type == 'values':
        for i in range(nrows):
            for j in range(ncols):
                handle.append(ax.text(j, i, str(A[i, j]), horizontalalignment='center', fontsize=fontsize))
    elif type == 'entries':
        for i in range(nrows):
            for j in range(ncols):
                if isinstance(A[i,j], str):
                    handle.append(ax.text(j, i, A[i, j], horizontalalignment='center', fontsize=fontsize))
                    
                else:  
                    handle.append(ax.text(j+1, i+1, ' ', horizontalalignment='center', fontsize=fontsize))
    elif type == 'patch':
        for i in range(nrows):
            for j in range(ncols):
                handle.append(ax.add_patch(
                    plt.Rectangle([i-0.5, j-0.5],
                                  width=1., height=1.,
                                  color=(A[i, j])*np.array([1, 1, 1]))))
    elif type == 'colorpatch':
        for i in range(nrows):
            for j in range(ncols):
                handle.append(ax.add_patch(
                    plt.Rectangle([i-0.5, j-0.5],
                                  width=1., height=1.,
                                  color=np.array([A[i, j, 0],
                                                  A[i, j, 1],
                                                  A[i, j, 2]]))))
                
                
    if bracket_style == 'boxes':
        x_lim = np.array([-0.5, ncols-0.5])
        ax.set_xlim(x_lim)
        y_lim = np.array([-0.5, nrows-0.5])
        ax.set_ylim(y_lim)
        for i in range(nrows+1):
            ax.add_line(plt.axhline(y=i-.5, #xmin=-0.5, xmax=ncols-0.5, 
                 color=bracket_color))
        for j in range(ncols+1):
            ax.add_line(plt.axvline(x=j-.5, #ymin=-0.5, ymax=nrows-0.5, 
                 color=bracket_color))
    elif bracket_style == 'square':
        tick_length = 0.25
        ax.plot([x_lim[0]+tick_length,
                     x_lim[0], x_lim[0],
                     x_lim[0]+tick_length],
                    [y_lim[0], y_lim[0],
                     y_lim[1], y_lim[1]],
                    linewidth=bracket_width,
                    color=np.array(bracket_color))
        ax.plot([x_lim[1]-tick_length, x_lim[1],
                              x_lim[1], x_lim[1]-tick_length],
                             [y_lim[0], y_lim[0], y_lim[1],
                              y_lim[1]],
                             linewidth=bracket_width, color=np.array(bracket_color))
      
    if highlight:       
        h_row = highlight_row
        h_col = highlight_col
        if isinstance(h_row, str) and h_row == ':':
            h_row = [0, nrows]
        if isinstance(h_col, str) and h_col == ':':
            h_col = [0, ncols]
        if len(h_row) == 1:
            h_row = [h_row, h_row]
        if len(h_col) == 1:
            h_col = [h_col, h_col]
        h_col.sort()
        h_row.sort()
        ax.add_line(plt.Line2D([h_col[0]-0.5, h_col[0]-0.5,
                              h_col[1]+0.5, h_col[1]+0.5,
                              h_col[0]-0.5],
                             [h_row[0]-0.5, h_row[1]+0.5,
                              h_row[1]+0.5, h_row[0]-0.5,
                              h_row[0]-0.5], color=highlight_color,
                               linewidth=highlight_width))
                    
    if zoom:      
        z_row = zoom_row
        z_col = zoom_col
        if isinstance(z_row, str) and z_row == ':':
            z_row = [1, nrows]
        if isinstance(z_col, str) and z_col == ':':
            z_col = [1, ncols]
        if len(z_row) == 1:
            z_row = [z_row, z_row]
        if len(z_col) == 1:
            z_col = [z_col, z_col]
        z_col.sort()
        z_row.sort()
        x_lim = [z_col[0]-0.5, z_col[1]+0.5]
        y_lim = [z_row[0]-0.5, z_row[1]+0.5]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis() #axis ij, axis equal, axis off

             
    return handle 


def base_plot(K, ind=[0, 1], ax=None,
              contour_color=[0., 0., 1],
              contour_style='-',
              contour_size=4,
              contour_markersize=4,
              contour_marker='x',
              fontsize=20):
    """
    % BASEPLOT Plot the contour of the covariance.
    % FORMAT
    % DESC creates the basic plot.
    % """

    blackcolor = [0,0,0]
    if ax is None:
        ax = plt.gca()
    v, U = np.linalg.eig(K[ind][:, ind])
    r = np.sqrt(v)
    theta = np.linspace(0, 2*np.pi, 200)[:, None]
    xy = np.dot(np.concatenate([r[0]*np.sin(theta), r[1]*np.cos(theta)], axis=1),U.T)
    cont = plt.Line2D(xy[:, 0], xy[:, 1],
                      linewidth=contour_size,
                      linestyle=contour_style,
                      color=contour_color)
    cent = plt.Line2D([0.], [0.],
                      marker=contour_marker,
                      color=contour_color,
                      linewidth=contour_size,
                      markersize=contour_markersize)

    ax.add_line(cont)
    ax.add_line(cent)

    thandle = []
    thandle.append(ax.set_xlabel('$f_{' + str(ind[1]+1)+ '}$',
                   fontsize=fontsize))
    thandle.append(ax.set_ylabel('$f_{' + str(ind[0]+1)+ '}$',
                   fontsize=fontsize))
    
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    x_lim = [-1.5, 1.5]
    y_lim = [-1.5, 1.5]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    ax.add_line(plt.Line2D(x_lim, [0, 0], color=blackcolor))
    ax.add_line(plt.Line2D([0, 0], y_lim, color=blackcolor))

    ax.set_aspect('equal')
    
    return cont, thandle, cent 


def prob_diagram(diagrams='../diagrams'):
    "Plot a diagram demonstrating marginal and joint probabilities."
    marg = 0.05 # Distance between lines and boxes
    indent = 0.1 # indent of n indicators
    axis_indent = 0.3 # Axis indent.

    x = np.random.randn(100, 1)+4
    y = np.random.randn(100, 1)+2.5

    fig, ax =plt.subplots(figsize=big_figsize)

    # Basic plot set up.    
    a = ax.plot(x, y, 'x', color = [1, 0, 0])
    plt.axis('off')
    ax.set_xlim([0-2*marg, 6+2*marg])
    ax.set_ylim([0-2*marg, 4+2*marg])
    #ax.set_visible(False)
    for i in range(7):
        ax.plot([i, i], [0, 5], color=[0, 0, 0])
    for i in range(5):
        ax.plot([0, 7], [i, i], color=[0, 0, 0])

    for i in range(1, 5):
        ax.text(-axis_indent, i-.5, str(i), horizontalalignment='center', fontsize=20)

    for i in range(1,7):
        ax.text(i-0.5, -axis_indent, str(i), horizontalalignment='center', fontsize=20)

    # Box for y=4
    ax.plot([-marg, 6+marg, 6+marg, -marg, -marg], [3-marg, 3-marg, 4+marg, 4+marg, 3-marg], linestyle=':', linewidth=2, color=[1, 0, 0])
    ax.text(0.5, 4-indent, '$n_{Y=4}$', horizontalalignment='center', fontsize=20)

    # Box for x=5
    ax.plot([4-marg, 5+marg, 5+marg, 4-marg, 4-marg], [-marg, -marg, 4+marg, 4+marg, -marg], linestyle='--', linewidth=2, color=[1, 0, 0])
    ax.text(4.5, 4-indent, '$n_{X=5}$', horizontalalignment='center', fontsize=20)

    # Box for x=3, y=4
    ax.plot([2-2*marg, 3+2*marg, 3+2*marg, 2-2*marg, 2-2*marg], [3-2*marg, 3-2*marg, 4+2*marg, 4+2*marg, 3-2*marg], linestyle='--', linewidth=2, color=[1, 0, 1])
    ax.text(2.5, 4-indent, '$n_{X=3, Y=4}$', horizontalalignment='center', fontsize=20)


    plt.text(1.5, 0.5, '$N$ crosses total', horizontalalignment='center', fontsize=20);

    plt.text(3, -2*axis_indent, '$X$', fontsize=20)
    plt.text(-2*axis_indent, 2, '$Y$', fontsize=20)
    #ylabel('\variableTwo')

    mlai.write_figure(os.path.join(diagrams, 'prob_diagram.svg'), transparent=True)



def hyperplane_coordinates(w, b, plot_limits):
    """Helper function for plotting the decision boundary of the perceptron."""
    if abs(w[1])>abs(w[0]):
        # If w[1]>w[0] in absolute value, plane is likely to be leaving tops of plot.
        x0 = plot_limits['x']
        x1 = -(b + x0*w[0])/w[1]
    else:
        # otherwise plane is likely to be leaving sides of plot.
        x1 = plot_limits['y']
        x0 = -(b + x1*w[1])/w[0]
    return x0, x1

def init_perceptron(f, ax, x_plus, x_minus, w, b, fontsize=18):
    """Initialise a plot for showing the perceptron decision boundary."""

    h = {}

    ax[0].set_aspect('equal')
    # Plot the data again
    ax[0].plot(x_plus[:, 0], x_plus[:, 1], 'rx')
    ax[0].plot(x_minus[:, 0], x_minus[:, 1], 'go')
    plot_limits = {}
    plot_limits['x'] = np.asarray(ax[0].get_xlim())
    plot_limits['y'] = np.asarray(ax[0].get_ylim())
    x0, x1 = hyperplane_coordinates(w, b, plot_limits)
    strt = -b/w[1]

    norm = w[0]*w[0] + w[1]*w[1]
    offset0 = -w[0]/norm*b
    offset1 = -w[1]/norm*b
    h['arrow'] = ax[0].arrow(offset0, offset1, offset0+w[0], offset1+w[1], head_width=0.2)
    # plot a line to represent the separating 'hyperplane'
    h['plane'], = ax[0].plot(x0, x1, 'b-')
    ax[0].set_xlim(plot_limits['x'])
    ax[0].set_ylim(plot_limits['y'])
    ax[0].set_xlabel('$x_0$', fontsize=fontsize)
    ax[0].set_ylabel('$x_1$', fontsize=fontsize)
    h['iter'] = ax[0].set_title('Update 0')
    
    bins = 15
    f_minus = np.dot(x_minus, w)
    f_plus = np.dot(x_plus, w)
    ax[1].hist(f_plus, bins, alpha=0.5, label='+1', color='r')
    ax[1].hist(f_minus, bins, alpha=0.5, label='-1', color='g')
    ax[1].legend(loc='upper right')
    return h

def update_perceptron(h, f, ax, x_plus, x_minus, i, w, b):
    """Update plots after decision boundary has changed."""
    # Helper function for updating plots
    # Re-plot the hyper plane 
    plot_limits = {}
    plot_limits['x'] = np.asarray(ax[0].get_xlim())
    plot_limits['y'] = np.asarray(ax[0].get_ylim())
    x0, x1 = hyperplane_coordinates(w, b, plot_limits)

    # Add arrow to represent hyperplane.
    h['arrow'].remove()
    del(h['arrow'])
    norm = (w[0]*w[0] + w[1]*w[1])
    offset0 = -w[0]/norm*b
    offset1 = -w[1]/norm*b
    h['arrow'] = ax[0].arrow(offset0, offset1, offset0+w[0], offset1+w[1], head_width=0.2)
    
    h['plane'].set_xdata(x0)
    h['plane'].set_ydata(x1)

    h['iter'].set_text('Update ' + str(i))
    ax[1].cla()
    bins = 15
    f_minus = np.dot(x_minus, w)
    f_plus = np.dot(x_plus, w)
    ax[1].hist(f_plus, bins, alpha=0.5, label='+1', color='r')
    ax[1].hist(f_minus, bins, alpha=0.5, label='-1', color='g')
    ax[1].legend(loc='upper right')

    IPython.display.display(f)
    IPython.display.clear_output(wait=True)
    return h

def contour_error(x, y, m_center, c_center, samps=100, width=6.):
    "Compute the error on a grid as a function of m and c."
    # create an array of linearly separated values around m_true
    m_vals = np.linspace(m_center-width/2., m_center+width/2., samps) 
    # create an array of linearly separated values ae
    c_vals = np.linspace(c_center-width/2., c_center+width/2., samps) 
    m_grid, c_grid = np.meshgrid(m_vals, c_vals)
    E_grid = np.zeros((samps, samps))
    for i in range(samps):
        for j in range(samps):
            E_grid[i, j] = ((y - m_grid[i, j]*x - c_grid[i, j])**2).sum()
    return m_vals, c_vals, E_grid
    
def regression_contour(f, ax, m_vals, c_vals, E_grid):
    "Regression contour plot."
    hcont = ax.contour(m_vals, c_vals, E_grid, levels=[0, 0.5, 1, 2, 4, 8, 16, 32, 64]) # this makes the contour plot 
    plt.clabel(hcont, inline=1, fontsize=15) # this labels the contours.

    ax.set_xlabel('$m$', fontsize=25)
    ax.set_ylabel('$c$', fontsize=25)

def init_regression(f, ax, x, y, m_vals, c_vals, E_grid, m_star, c_star, fontsize=20):
    """Function to plot the initial regression fit and the error surface."""
    h = {}
    levels=[0, 0.5, 1, 2, 4, 8, 16, 32, 64]
    h['cont'] = ax[0].contour(m_vals, c_vals, E_grid, levels=levels) # this makes the contour plot on axes 0.
    plt.clabel(h['cont'], inline=1, fontsize=15)
    ax[0].set_xlabel('$m$', fontsize=fontsize)
    ax[0].set_ylabel('$c$', fontsize=fontsize)
    h['msg'] = ax[0].set_title('Error Function', fontsize=fontsize)

    # Set up plot
    h['data'], = ax[1].plot(x, y, 'r.', markersize=10)
    ax[1].set_xlabel('$x$', fontsize=fontsize)
    ax[1].set_ylabel('$y$', fontsize=fontsize)
    ax[1].set_ylim((-9, -1)) # set the y limits of the plot fixed
    ax[1].set_title('Best Fit', fontsize=fontsize)

    # Plot the current estimate of the best fit line
    x_plot = np.asarray(ax[1].get_xlim()) # get the x limits of the plot for plotting the current best line fit.
    y_plot = m_star*x_plot + c_star
    h['fit'], = ax[1].plot(x_plot, y_plot, 'b-', linewidth=3)
    return h

def update_regression(h, f, ax, m_star, c_star, iteration):
    """Update the regression plot with the latest fit and position in error space."""
    ax[0].plot(m_star, c_star, 'g*')
    x_plot = np.asarray(ax[1].get_xlim()) # get the x limits of the plot for plo
    y_plot = m_star*x_plot + c_star
    
    # show the current status on the plot of the data
    h['fit'].set_ydata(y_plot)
    h['msg'].set_text('Iteration '+str(iteration))
    IPython.display.display(f)
    IPython.display.clear_output(wait=True)
    return h

def regression_contour_fit(x, y, learn_rate=0.01, m_center=1.4, c_center=-3.1, m_star = 0.0, c_star = -5.0, max_iters=1000, diagrams='../diagrams'):
    "Plot an evolving contour plot of regression optimisation."
    m_vals, c_vals, E_grid = contour_error(x, y, m_center, c_center, samps=100)

    f, ax = plt.subplots(1, 2, figsize=two_figsize) # this is to create 'side by side axes'
    # first let's plot the error surface
    handle = init_regression(f, ax, x, y, m_vals, c_vals, E_grid, m_star, c_star)
    mlai.write_figure(os.path.join(diagrams, 'regression_contour_fit000.svg'), transparent=True)

    count=0
    for i in range(max_iters): # do max_iters iterations
        # compute the gradients
        c_grad = -2*(y-m_star*x - c_star).sum()
        m_grad = -2*(x*(y-m_star*x - c_star)).sum()

        # update the parameters
        m_star = m_star - learn_rate*m_grad
        c_star = c_star - learn_rate*c_grad
        # update the location of our current best guess on the contour plot
        if i<10 or ((i<100 and not i % 10) or (i<1000 and not i % 100)): 
            handle = update_regression(handle, f, ax, m_star, c_star, i)
            count+=1
            mlai.write_figure(os.path.join(diagrams, 'regression_contour_fit{count:0>3}.svg').format(count=count))        
    return count

def regression_contour_sgd(x, y, learn_rate=0.01, m_center=1.4, c_center=-3.1, m_star = 0.0, c_star = -5.0, max_iters=4000, diagrams='../diagrams'):
    "Plot evolution of the solution of linear regression via SGD."
    m_vals, c_vals, E_grid = contour_error(x, y, m_center, c_center, samps=100)

    f, ax = plt.subplots(1, 2, figsize=two_figsize) # this is to create 'side by side axes'
    handle = init_regression(f, ax, x, y, m_vals, c_vals, E_grid, m_star, c_star)
    count=0
    mlai.write_figure(os.path.join(diagrams, 'regression_sgd_contour_fit{count:0>3}.svg').format(count=count))
    for i in range(max_iters): # do max_iters iterations (parameter updates)
        # choose a random point
        index = np.random.randint(x.shape[0]-1)

        # update m
        m_star = m_star + 2*learn_rate*(x[index]*(y[index]-m_star*x[index] - c_star))
        # update c
        c_star = c_star + 2*learn_rate*(y[index]-m_star*x[index] - c_star)

        if i<10 or ((i<100 and not i % 10) or (not i % 100)): 
            handle = update_regression(handle, f, ax, m_star, c_star, i)
            count+=1
            mlai.write_figure(os.path.join(diagrams, 'regression_sgd_contour_fit{count:0>3}.svg').format(count=count))
    return count

#################### Session 3 ####################
def over_determined_system(diagrams='../diagrams'):
    """Visualize what happens in an over determined system with linear regression."""
    x = np.array([1, 3])
    y = np.array([3, 1])

    xvals = np.linspace(0, 5, 2);

    m = (y[1]-y[0])/(x[1]-x[0]);
    c = y[0]-m*x[0];

    yvals = m*xvals+c;
    xvals = np.linspace(0, 5, 2);

    m = (y[1]-y[0])/(x[1]-x[0]);
    c = y[0]-m*x[0];

    yvals = m*xvals+c;

    ylim = np.array([0, 5])
    xlim = np.array([0, 5])

    f, ax = plt.subplots(1,1,figsize=one_figsize)
    a = ax.plot(xvals, yvals, '-', linewidth=3);

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.xlabel('$x$', fontsize=30)
    plt.ylabel('$y$',fontsize=30)
    plt.text(4, 4, '$y=mx+c$',  horizontalalignment='center', verticalalignment='bottom', fontsize=30)
    mlai.write_figure(os.path.join(diagrams, 'over_determined_system001.svg'), transparent=True)
    ctext = ax.text(0.15, c+0.15, '$c$',  horizontalalignment='center', verticalalignment='bottom', fontsize=20)
    xl = np.array([1.5, 2.5])
    yl = xl*m + c;
    mhand = ax.plot([xl[0], xl[1]], [yl.min(), yl.min()], color=[0, 0, 0])
    mhand2 = ax.plot([xl.min(), xl.min()], [yl[0], yl[1]], color=[0, 0, 0])
    mtext = ax.text(xl.mean(), yl.min()-0.2, '$m$',  horizontalalignment='center', verticalalignment='bottom',fontsize=20);
    mlai.write_figure(os.path.join(diagrams, 'over_determined_system002.svg'), transparent=True)

    a2 = ax.plot(x, y, '.', markersize=20, linewidth=3, color=[1, 0, 0])
    mlai.write_figure(os.path.join(diagrams, 'over_determined_system003.svg'), transparent=True)

    xs = 2
    ys = m*xs + c + 0.3

    ast = ax.plot(xs, ys, '.', markersize=20, linewidth=3, color=[0, 1, 0])
    mlai.write_figure(os.path.join(diagrams, 'over_determined_system004.svg'), transparent=True)


    m = (y[1]-ys)/(x[1]-xs);
    c = ys-m*xs;
    yvals = m*xvals+c;

    for i in a:
        i.set_visible(False)
    for i in mhand:
        i.set_visible(False)
    for i in mhand2:
        i.set_visible(False)
    mtext.set_visible(False)
    ctext.set_visible(False)
    a3 = ax.plot(xvals, yvals, '-', linewidth=2, color=[0, 0, 1])
    for i in ast:
        i.set_color([1, 0, 0])
    mlai.write_figure(os.path.join(diagrams, 'over_determined_system005.svg'), transparent=True)

    m = (ys-y[0])/(xs-x[0])
    c = y[0]-m*x[0]
    yvals = m*xvals+c

    for i in a3:
        i.set_visible(False)
    a4 = ax.plot(xvals, yvals, '-', linewidth=2, color=[0, 0, 1]);
    for i in ast:
        i.set_color([1, 0, 0])
    mlai.write_figure(os.path.join(diagrams, 'over_determined_system006.svg'), transparent=True)
    for i in a:
        i.set_visible(True)
    for i in a3:
        i.set_visible(True)
    mlai.write_figure(os.path.join(diagrams, 'over_determined_system007.svg'), transparent=True)

def gaussian_of_height(diagrams='../diagrams'):
    "Gaussian density representing heights."
    h = np.linspace(0, 2.5, 1000)
    sigma2 = 0.0225
    mu = 1.7
    p = 1./np.sqrt(2*np.pi*sigma2)*np.exp(-(h-mu)**2/(2*sigma2**2))
    f2, ax2 = plt.subplots(figsize=wide_figsize)
    ax2.plot(h, p, 'b-', linewidth=3)
    ylim = (0, 3)
    ax2.vlines(mu, ylim[0], ylim[1], colors='r', linewidth=3)
    ax2.set_ylim(ylim)
    ax2.set_xlim(1.4, 2.0)
    ax2.set_xlabel('$h/m$', fontsize=20)
    ax2.set_ylabel('$p(h|\mu, \sigma^2)$', fontsize = 20)
    mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'gaussian_of_height.svg'), transparent=True)
    
#################### Session 5 ####################

def marathon_fit(model, value, param_name, param_range, xlim, fig, ax, x_val=None, y_val=None, objective=None, diagrams='../diagrams', fontsize=20, objective_ylim=None, prefix='olympic', title=None, png_plot=False, samps=130):
    "Plot fit of the marathon data alongside error."
    if title is None:
        title = model.objective_name
        
    ax[0].cla()
    ax[0].plot(model.X, model.y, 'o', color=[1, 0, 0], markersize=6, linewidth=3)
    if x_val is not None and y_val is not None:
        ax[0].plot(x_val, y_val, 'o', color=[0, 1, 0], markersize=6, linewidth=3)
        
    ylim = ax[0].get_ylim()

    x_pred = np.linspace(xlim[0], xlim[1], samps)[:, None]
    y_pred, y_var = model.predict(x_pred)
    
    ax[0].plot(x_pred, y_pred, color=[0, 0, 1], linewidth=2)
    if y_var is not None:
        y_err = np.sqrt(y_var)*2
        ax[0].plot(x_pred, y_pred + y_err, '--', color=[0, 0, 1], linewidth=1)
        ax[0].plot(x_pred, y_pred - y_err, '--', color=[0, 0, 1], linewidth=1)
        
    #ax[0].set_xlabel('year', fontsize=fontsize)
    ax[0].set_ylim(ylim)
    plt.sca(ax[0])

    xlim = ax[0].get_xlim()

    if objective is not None:
        ax[1].cla()
        params = range(*param_range)
        for name, vals in objective.items():
            ax[1].plot(np.array(params), vals, 'o', color=[1, 0, 0], markersize=6, linewidth=3)
        if len(param_range)>2:
            xlow = param_range[0]-param_range[2]
            xhigh = param_range[1]
        else:
            xlow = param_range[0]-1
            xhigh = param_range[1]
        ax[1].set_xlim((xlow, xhigh))
        ax[1].set_ylim(objective_ylim)
        ax[1].set_xlabel(param_name.replace('_', ' '), fontsize=fontsize)
        if title is not None:
            ax[1].set_title(title, fontsize=fontsize)

    filename = '{prefix}_{name}_{param_name}{value:0>3}'.format(prefix=prefix, name=model.name, param_name=param_name, value=value)
    mlai.write_figure(os.path.join(diagrams, filename + '.svg'), transparent=True)
    if png_plot:
        mlai.write_figure(os.path.join(directory, filename + '.png'), transparent=True)



def rmse_fit(x, y, param_name, param_range, model=mlai.LM, plot_objectives={'RMSE':mlai.MapModel.rmse}, objective_ylim=None, xlim=None, plot_fit=marathon_fit, diagrams='../diagrams', **kwargs):
    "Fit a model and show RMSE error"
    f, ax = plt.subplots(1, 2, figsize=two_figsize)
    num_data = x.shape[0]
    
    params = range(*param_range)

    count = 0
    for param in params:
        kwargs[param_name] = param
        m = model(x, y, **kwargs)
        m.fit()
        # compute appropriate objective. 
        for name, plot_objective in plot_objectives.items():
            obj[name][count] = plot_objective(m)
            
        plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                 objective=obj, objective_ylim=objective_ylim,
                 fig=f, ax=ax, diagrams=diagrams)
        count += 1


def holdout_fit(x, y, param_name, param_range, model=mlai.LM, val_start=20,
                objective_ylim=None, xlim=None, plot_fit=marathon_fit,
                permute=True, prefix='olympic_val', diagrams='../diagrams', **kwargs):
    "Fit a model and show holdout error."

    f, ax = plt.subplots(1, 2, figsize=two_figsize)

    num_data = x.shape[0]

    if permute:
        perm = np.random.permutation(num_data)
        x_tr = x[perm[:val_start], :]
        x_val = x[perm[val_start:], :]
        y_tr = y[perm[:val_start], :]
        y_val = y[perm[val_start:], :]
    else:
        x_tr = x[:val_start, :]
        x_val = x[val_start:, :]
        y_tr = y[:val_start, :]
        y_val = y[val_start:, :]
    num_val_data = x_val.shape[0]

    params = range(*param_range)
    ll = np.array([np.nan]*len(params))
    ss = np.array([np.nan]*len(params))
    ss_val = np.array([np.nan]*len(params))
    count = 0
    for param in params:    
        kwargs[param_name] = param
        m = model(x_tr, y_tr, **kwargs)
        m.fit()
        f_val, _ = m.predict(x_val)
        ss[count] = m.objective()
        ss_val[count] = ((y_val-f_val)**2).mean() 
        ll[count] = m.log_likelihood()
        plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                 objective=np.sqrt(ss_val), objective_ylim=objective_ylim,
                 fig=f, ax=ax, prefix=prefix,
                 title="Hold Out Validation",
                 x_val=x_val, y_val=y_val, diagrams=diagrams)
        count+=1

def loo_fit(x, y, param_name, param_range, model=mlai.LM, objective_ylim=None, 
            xlim=None, plot_fit=marathon_fit, prefix='olympic_loo', diagrams='../diagrams', **kwargs):
    "Fit a model and show leave one out error"
    f, ax = plt.subplots(1, 2, figsize=two_figsize)


    num_data = x.shape[0]
    num_parts = num_data
    partitions = []
    for part in range(num_parts):
        train_ind = list(range(part))
        train_ind.extend(range(part+1,num_data))
        val_ind = [part]
        partitions.append((train_ind, val_ind))

        params = range(*param_range)        
        ll = np.array([np.nan]*len(params))
        ss = np.array([np.nan]*len(params))
        ss_val = np.array([np.nan]*len(params))
        count = 0
        for param in params:
            kwargs[param_name] = param
            ss_temp = 0.
            ll_temp = 0.
            ss_val_temp = 0.
            for part, (train_ind, val_ind) in enumerate(partitions):
                x_tr = x[train_ind, :]
                x_val = x[val_ind, :]
                y_tr = y[train_ind, :]
                y_val = y[val_ind, :]
                num_val_data = x_val.shape[0]
                m = model(x_tr, y_tr, **kwargs)
                m.fit()
                ss_temp = m.objective()
                ll_temp = m.log_likelihood()
                f_val, _ = m.predict(x_val)
                ss_val_temp += ((y_val-f_val)**2).mean() 
                plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                         objective=np.sqrt(ss_val), objective_ylim=objective_ylim,
                         fig=f, ax=ax, prefix='olympic_loo{part:0>3}'.format(part=part),
                         x_val=x_val, y_val=y_val, diagrams=diagrams)
            ss[count] = ss_temp/(num_parts)
            ll[count] = ll_temp/(num_parts)
            ss_val[count] = ss_val_temp/(num_parts)
            ax[1].cla()
            plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                     objective=np.sqrt(ss_val), objective_ylim=objective_ylim,
                     fig=f, ax=ax, prefix='olympic_loo{part:0>3}'.format(part=len(partitions)),
                     title="Leave One Out Validation",
                     x_val=x_val, y_val=y_val, diagrams=diagrams)
            count+=1


def cv_fit(x, y, param_name, param_range, model=mlai.LM, objective_ylim=None, 
               xlim=None, plot_fit=marathon_fit, num_parts=5, diagrams='../diagrams', **kwargs):
    f, ax = plt.subplots(1, 2, figsize=two_figsize)
    num_data = x.shape[0]
    partitions = []
    ind = list(np.random.permutation(num_data))
    start = 0
    for part in range(num_parts):
        end = round((float(num_data)/num_parts)*(part+1))
        train_ind = ind[:start]
        train_ind.extend(ind[end:])
        val_ind = ind[start:end]
        partitions.append((train_ind, val_ind))
        start = end

    params = range(*param_range)
    ll = np.array([np.nan]*len(params))
    ss = np.array([np.nan]*len(params))
    ss_val = np.array([np.nan]*len(params))
    count = 0
    for param in params:
        ss_val_temp = 0.
        ll_temp = 0.
        ss_temp = 0.
        kwargs[param_name] = param
        for part, (train_ind, val_ind) in enumerate(partitions):
            x_tr = x[train_ind, :]
            x_val = x[val_ind, :]
            y_tr = y[train_ind, :]
            y_val = y[val_ind, :]
            num_val_data = x_val.shape[0]

            m = model(x_tr, y_tr, **kwargs)
            m.fit()
            ss_temp += m.objective()
            ll_temp += m.log_likelihood()
            f_val, _ = m.predict(x_val)
            ss_val_temp += ((y_val-f_val)**2).mean() 
            plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                     objective=np.sqrt(ss_val), objective_ylim=objective_ylim,
                     fig=f, ax=ax, prefix='olympic_{num_parts}cv{part:0>2}'.format(num_parts=num_parts, part=part),
                     title='{num_parts}-fold Cross Validation'.format(num_parts=num_parts),
                     x_val=x_val, y_val=y_val, diagrams=diagrams)
        ss_val[count] = ss_val_temp/(num_parts)
        ss[count] = ss_temp/(num_parts)
        ll[count] = ll_temp/(num_parts)
        count+=1
        ax[1].cla()
        plot_fit(model=m, value=param, xlim=xlim, param_name=param_name, param_range=param_range,
                 objective=np.sqrt(ss_val), objective_ylim=objective_ylim,
                 fig=f, ax=ax,
                 prefix='olympic_{num_parts}cv{num_partitions:0>2}'.format(num_parts=num_parts, num_partitions=num_parts),
                 title='{num_parts}-fold Cross Validation'.format(num_parts=num_parts),
                 x_val=x_val, y_val=y_val, diagrams=diagrams)
            
#################### Session 6 ####################    

def under_determined_system(diagrams='../diagrams'):
    """Visualise what happens in an under determined system with linear regression."""
    x = 1.
    y = 3.
    fig, ax = plt.subplots(figsize=one_figsize)
    ax.plot(x, y, 'o', markersize=10, linewidth=3, color=[1., 0., 0.])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ylim = [0, 5]
    xlim = [0, 3]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$y$', fontsize=20)
    mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'under_determined_system000.svg'), transparent=True, frameon=True)

    xvals = np.linspace(0, 3, 2)[:, None]
    count=0
    for i in range(100):
        c = np.random.normal(size=(1,1))*2
        m = (y - c)/x
        yvals = m*xvals+c
        ax.plot(xvals, yvals, '-', linewidth=2, color=[0., 0., 1.])
        if i < 9 or i == 100:
            count += 1
            mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'under_determined_system{count:0>3}.svg').format(count=count), transparent=True, frameon=True)


def bayes_update(diagrams='../diagrams'):
    "Visualise the updating of a posterior of Bayesian inference for a Gaussian lieklihood."""
    fig, ax = plt.subplots(figsize=two_figsize)
    num_points = 1000
    x_max = 6
    x_min = -1

    y = np.array([[1.]])
    prior_mean = np.array([[0.]])
    prior_var = np.array([[.1]])

    noise = mlai.Gaussian(offset=np.array([0.6]), scale=np.array(np.sqrt(0.05)))


    f = np.linspace(x_min, x_max, num_points)[:, None]
    ln_prior_curve = -0.5*(np.log(2*np.pi*prior_var) + (f-prior_mean)*(f-prior_mean)/prior_var)
    ln_likelihood_curve = np.zeros(ln_prior_curve.shape)
    for i in range(num_points):
        ln_likelihood_curve[i] = noise.log_likelihood(f[i][None, :], 
                                                      np.array([[np.finfo(float).eps]]), 
                                                      y)
    ln_marginal_likelihood = noise.log_likelihood(prior_mean, prior_var, y);

    prior_curve = np.exp(ln_prior_curve) 
    likelihood_curve = np.exp(ln_likelihood_curve)
    marginal_curve = np.exp(ln_marginal_likelihood)

    ln_posterior_curve = ln_likelihood_curve + ln_prior_curve - ln_marginal_likelihood
    posterior_curve = np.exp(ln_posterior_curve)

    g, dlnZ_dvs = noise.grad_vals(prior_mean, prior_var, y)

    nu = g*g - 2*dlnZ_dvs

    approx_var = prior_var - prior_var*prior_var*nu
    approx_mean = prior_mean + prior_var*g

    ln_approx_curve = -0.5*np.log(2*np.pi*approx_var)-0.5*(f-approx_mean)*(f-approx_mean)/approx_var

    approx_curve = np.exp(ln_approx_curve)
    noise
    xlim = [x_min, x_max] 
    ylim = [0, np.vstack([approx_curve, likelihood_curve, 
                          posterior_curve, prior_curve]).max()*1.1]

    fig, ax = plt.subplots(figsize=two_figsize)

    ax.set_xlim(xlim)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(ylim)

    ax.vlines(xlim[0], ylim[0], ylim[1], color=[0., 0., 0.]) 
    ax.hlines(ylim[0], xlim[0], xlim[1], color=[0., 0., 0.]) 

    ax.plot(f, prior_curve, color=[1, 0., 0.], linewidth=3)
    ax.text(3.5, 2, '$p(c) = \mathcal{N}(c|0, \\alpha_1)$', horizontalalignment='center', fontsize=20) 
    mlai.write_figure(os.path.join(diagrams, 'dem_gaussian001.svg'), transparent=True)

    ax.plot(f, likelihood_curve, color=[0, 0, 1], linewidth=3)
    ax.text(3.5, 1.5,'$p(y|m, c, x, \\sigma^2)=\mathcal{N}(y|mx+c,\\sigma^2)$', horizontalalignment='center', fontsize=20) 
    mlai.write_figure(os.path.join(diagrams, 'dem_gaussian002.svg'), transparent=True)

    ax.plot(f, posterior_curve, color=[1, 0, 1], linewidth=3)
    ax.text(3.5, 1, '$p(c|y, m, x, \\sigma^2)=$', horizontalalignment='center', fontsize=20) 
    plt.text(3.5, 0.65, '$\mathcal{N}\\left(c|\\frac{y-mx}{1+\\sigma^2\\alpha_1},(\\sigma^{-2}+\\alpha_1^{-1})^{-1}\\right)$', horizontalalignment='center', fontsize=20)
    mlai.write_figure(os.path.join(diagrams, 'dem_gaussian003.svg'), transparent=True)

def height_weight(h=None, w=None, muh=1.7, varh=0.0225, muw=75, varw=36, diagrams='../diagrams'):
    "Plot height and weight as Gaussians."
    if h is None:
        h = np.linspace(1.25, 2.15, 100)[:, None]
    if w is None:
        w = np.linspace(55, 95, 100)[:, None]

    ph = 1/np.sqrt(tau*varh)*np.exp(-1/(2*varh)*(h - muh)**2)
    pw = 1/np.sqrt(tau*varw)*np.exp(-1/(2*varw)*(w - muw)**2)

    fig, ax = plt.subplots(1, 2, figsize=two_figsize)

    height(ax[0], h, ph)

    weight(ax[1], w, pw)
    mlai.write_figure(os.path.join(diagrams, 'height_weight_gaussian.svg'), transparent=True)

def independent_height_weight(h=None, w=None, muh=1.7, varh=0.0225, muw=75, varw=36, num_samps=20, diagrams='../diagrams'):
    "Plot independent Gaussians of height and weight."
    if h is None:
        h = np.linspace(1.25, 2.15, 100)[:, None]
    if w is None:
        w = np.linspace(55, 95, 100)[:, None]

    ph = 1/np.sqrt(tau*varh)*np.exp(-1/(2*varh)*(h - muh)**2)
    pw = 1/np.sqrt(tau*varw)*np.exp(-1/(2*varw)*(w - muw)**2)
    
    fig, axs = plt.subplots(2, 4, figsize=two_figsize)
    for a in axs.flatten():
        a.set_axis_off()
    ax=[]
    ax.append(plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2))
    ax.append(plt.subplot2grid((2,4), (0,3)))
    ax.append(plt.subplot2grid((2,4), (1,3)))

    ax[0].plot(muh, muw, 'x', color=[1., 0., 1.], markersize=5., linewidth=3)
    theta = np.linspace(0, tau, 100)
    xel = np.sin(theta)*np.sqrt(varh) + muh
    yel = np.cos(theta)*np.sqrt(varw) + muw
    ax[0].plot(xel, yel, '-', color=[1., 0., 1.], linewidth=3)
    ax[0].set_xlim([h.min(), h.max()])
    ax[0].set_ylim([w.min()+10, w.max()-10])
    ax[0].set_yticks([65, 75, 85])
    ax[0].set_xticks([1.25, 1.7, 2.15])
    ax[0].set_xlabel('$h/m$', fontsize=20)
    ax[0].set_ylabel('$w/kg$', fontsize=20)

    ylim = ax[0].get_ylim()
    xlim = ax[0].get_xlim()
    ax[0].vlines(xlim[0], ylim[0], ylim[1], color=[0.,0.,0.])
    ax[0].hlines(ylim[0], xlim[0], xlim[1], color=[0., 0., 0.])

    height(ax[1], h, ph)
    weight(ax[2], w, pw)
    count = 0
    for i in range(num_samps):
        hval = np.random.normal(size=(1,1))*np.sqrt(varh) + muh
        wval = np.random.normal(size=(1,1))*np.sqrt(varw) + muw
        a1 = ax[1].plot(hval, 0.1, marker='o', linewidth=3, color=[1., 0., 0.])
        #mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'independent_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        #count+=1
        a2 = ax[2].plot(wval, 0.002, marker='o', linewidth=3, color=[1., 0., 0.])
        #mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'independent_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        #count+=1
        a0 = ax[0].plot(hval, wval, marker='o', linewidth=3, color=[1., 0., 0.])
        mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'independent_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        count+=1

        a0[0].set(color=[0.,0.,0.])
        a1[0].set(color=[0.,0.,0.])
        a2[0].set(color=[0.,0.,0.])
        
        #mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'independent_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        #count+=1

def correlated_height_weight(h=None, w=None, muh=1.7, varh=0.0225, muw=75, varw=36, num_samps=20, diagrams='../diagrams'):
    "Plot correlated Gaussian distributions of height and weight."
    if h is None:
        h = np.linspace(1.25, 2.15, 100)[:, None]
    if w is None:
        w = np.linspace(55, 95, 100)[:, None]

    ph = 1/np.sqrt(tau*varh)*np.exp(-1/(2*varh)*(h - muh)**2)
    pw = 1/np.sqrt(tau*varw)*np.exp(-1/(2*varw)*(w - muw)**2)

    fig, axs = plt.subplots(2, 4, figsize=two_figsize)
    for a in axs.flatten():
        a.set_axis_off()
    ax=[]
    ax.append(plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2))
    ax.append(plt.subplot2grid((2,4), (0,3)))
    ax.append(plt.subplot2grid((2,4), (1,3)))

    covMat = np.asarray([[1, 0.995], [0.995, 1]])
    fact = np.asarray([[np.sqrt(varh), 0], [0, np.sqrt(varw)]])
    covMat = np.dot(np.dot(fact,covMat), fact)
    _, R = np.linalg.eig(covMat)

    ax[0].plot(muh, muw, 'x', color=[1., 0., 1.], markersize=5, linewidth=3)
    theta = np.linspace(0, tau, 100)
    xel = np.sin(theta)*np.sqrt(varh)
    yel = np.cos(theta)*np.sqrt(varw)
    vals = np.dot(R,np.vstack([xel, yel]))
    ax[0].plot(vals[0, :]+muh, vals[1, :]+muw, '-', color=[1., 0., 1.], linewidth=3)
    ax[0].set_xlim([h.min(), h.max()])
    ax[0].set_ylim([w.min()+10, w.max()-10])
    ax[0].set_yticks([65, 75, 85])
    ax[0].set_xticks([1.25, 1.7, 2.15])
    ax[0].set_xlabel('$h/m$', fontsize=20)
    ax[0].set_ylabel('$w/kg$', fontsize=20)

    height(ax[1], h, ph)
    weight(ax[2], w, pw)
    count = 0
    for i in range(num_samps):
        vec_s = np.dot(np.dot(R,fact),np.random.normal(size=(2,1)))
        hval = vec_s[0] + muh
        wval = vec_s[1] + muw
        a1 = ax[1].plot(hval, 0.1, marker='o', linewidth=3, color=[1., 0., 0.])
        #mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'correlated_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        a2 = ax[2].plot(wval, 0.002, marker='o', linewidth=3, color=[1., 0., 0.])
        #count+=1
        #mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'correlated_height_weight{count:0>3}.svg').format(count=count), transparent=True)

        a0 = ax[0].plot(hval, wval, marker='o', linewidth=3, color=[1., 0., 0.])
        #count+=1
        mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'correlated_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        #count+=1

        a0[0].set(color=[0.,0.,0.])
        a1[0].set(color=[0.,0.,0.])
        a2[0].set(color=[0.,0.,0.])

        #mlai.write_figure(figure=fig, filename=os.path.join(diagrams, 'correlated_height_weight{count:0>3}.svg').format(count=count), transparent=True)
        count+=1




#################### Session 11 ####################

def two_point_pred(K, f, x, ax=None, ind=[0, 1],
                        conditional_linestyle = '-',
                        conditional_linecolor = [1., 0., 0.],
                        conditional_size = 4,
                        fixed_linestyle = '-',
                        fixed_linecolor = [0., 1., 0.],
                        fixed_size = 4,stub=None, start=0, diagrams='../diagrams'):
    
    subK = K[ind][:, ind]
    f = f[ind]
    x = x[ind]

    if ax is None:
        ax = plt.gca()

    cont, t, cent = base_plot(K, ind, ax=ax)
    if stub is not None:
        mlai.write_figure(os.path.join(diagrams, '{stub}{start:0>3}.svg').format(stub=stub, start=start), transparent=True)

    x_lim = ax.get_xlim()
    cont2 = plt.Line2D([x_lim[0], x_lim[1]], [f[0], f[0]], linewidth=fixed_size, linestyle=fixed_linestyle, color=fixed_linecolor)
    ax.add_line(cont2)

    if stub is not None:
        mlai.write_figure(os.path.join(diagrams, '{stub}{start:0>3}.svg').format(stub=stub, start=start+1), transparent=True)

    # # Compute conditional mean and variance
    f2_mean = subK[0, 1]/subK[0, 0]*f[0]
    f2_var = subK[1, 1] - subK[0, 1]/subK[0, 0]*subK[0, 1]
    x_val = np.linspace(x_lim[0], x_lim[1], 200)
    pdf_val = 1/np.sqrt(2*np.pi*f2_var)*np.exp(-0.5*(x_val-f2_mean)*(x_val-f2_mean)/f2_var)
    pdf = plt.Line2D(x_val, pdf_val+f[0], linewidth=conditional_size, linestyle=conditional_linestyle, color=conditional_linecolor)
    ax.add_line(pdf)
    if stub is not None:
        mlai.write_figure(os.path.join(diagrams, '{stub}{start:0>3}.svg').format(stub=stub, start=start+2), transparent=True)
    
    obs = plt.Line2D([f[1]], [f[0]], linewidth=10, markersize=10, color=fixed_linecolor, marker='o')
    ax.add_line(obs)
    if stub is not None:
        mlai.write_figure(os.path.join(diagrams, '{stub}{start:0>3}.svg').format(stub=stub, start=start+3), transparent=True)
    

def kern_circular_sample(K, mu=None, filename=None, fig=None, num_samps=5, num_theta=200, diagrams='../diagrams'):

    """Make an animation of a circular sample from a covariance function."""

    n = K.shape[0]


    R1 = np.random.normal(size=(n, num_samps))
    U1 = np.dot(R1,np.diag(1/np.sqrt(np.sum(R1*R1, axis=0))))
    R2 = np.random.normal(size=(n, num_samps))
    R2 = R2 - np.dot(U1,np.diag(np.sum(R2*U1, axis=0)))
    R2 = np.dot(R2,np.diag(np.sqrt(np.sum(R1*R1, axis=0))/np.sqrt(np.sum(R2*R2, axis=0))))
    L = np.linalg.cholesky(K+np.diag(np.ones((n)))*1e-6)


    from matplotlib import animation
    x_lim = (0, 1)
    y_lim = (-2, 2)
    
    if fig is None:
        fig, _ = plt.subplots(figsize=one_figsize)
    rect = 0, 0, 1., 1.
    ax = fig.add_axes(rect)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    line = []
    for i in range(num_samps):
        l, = ax.plot([], [], lw=2)
        line.append(l)
        
    # initialization function: plot the background of each frame
    def init():
        for i in range(num_samps):
            line[i].set_data([], [])
        return line

    # animation function.  This is called sequentially
    def animate(i):
        theta = float(i)/num_theta*tau
        xc = np.cos(theta)
        yc = np.sin(theta)
        # generate 2d basis in t-d space
        coord = xc*R1 + yc*R2
        y = np.dot(L,coord)
        if mu is not None:
            y = y + mu
        x = np.linspace(0, 1, n)
        for i in range(num_samps):
            line[i].set_data(x, y[:, i])
        return line

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_theta, blit=True)
    if filename is not None:
        anim.save(os.path.join(diagrams, filename), writer='imagemagick', fps=30)


def covariance_func(x, kernel_function, x_cov=None, formula=None, shortname=None, longname=None, comment=None, diagrams='../diagrams', **args):
    """Write a slide on a given covariance matrix."""
    fig, ax = plt.subplots(figsize=one_figsize)
    hcolor = [1., 0., 1.]
    if x_cov is None:
        x_cov = x
    K = kernel_function(x_cov, x_cov, **args)
    obj = matrix(K, ax=ax, type='image', bracket_style='boxes')

    if shortname is not None:
        filename = shortname + '_covariance'
    else:
        filename = 'covariance'
    mlai.write_figure(os.path.join(diagrams, filename + '.svg'), transparent=True)

    ax.cla()
    kern_circular_sample(K, fig=fig, filename=filename + '.gif', diagrams=diagrams)

    out = '<h2>' + longname + ' Covariance</h2>'
    out += '\n\n'
    out += '<p><center>' + formula + '</center></p>'
    out += '<table>\n  <tr><td><img src="' + os.path.join(diagrams, filename) + '.svg"></td><td><img src="' + os.path.join(diagrams, filename) + '.gif"></td></tr>\n</table>'
    if comment is not None:
        out += '<p><center>' + comment + '</center></p>'
    fhand = open(os.path.join(diagrams, filename + '.html'), 'w')
    fhand.write(out)


def two_point_sample(kernel_function, diagrams='../diagrams', **args):
    """Make plots for the two data point sample example for explaining gaussian processes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(two_figsize))
    x = np.linspace(-1, 1, 25)[:, None]
    K = kernel_function(x, x, **args)
    obj = matrix(K, ax=ax[1], type='image')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    #fig.colorbar(mappable=obj, ax=ax[1])
    #ax[1].set_axis('off')
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample000.svg'), transparent=True)

    f = np.random.multivariate_normal(np.zeros(25), K, size=1)
    ax[0].plot(range(1, 26), f.flatten(), 'o', markersize=5, linewidth=3, color=[1., 0., 0.])
    ax[0].set_xticks(range(1, 26, 2))
    ax[0].set_yticks([-2, -1, 0, 1, 2])
    ylim = [-2, 2]
    xlim = [0, 26]
    ax[0].set_ylim(ylim)
    ax[0].set_xlim(xlim)
    ax[0].set_xlabel('$i$', fontsize=20)
    ax[0].set_ylabel('$f$', fontsize=20)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample001.svg'), transparent=True)

    ax[0].plot(np.array([1, 2]), [f[0,0], f[0,1]], 'o', markersize=10, linewidth=5, color=hcolor)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample002.svg'), transparent=True)
    #plt.Circle?

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor)
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample003.svg'), transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=5,
                     zoom=True,
                     zoom_row=[0, 9],
                     zoom_col=[0, 9])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample004.svg'), transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=6,
                     zoom=True,
                     zoom_row=[0, 4],
                     zoom_col=[0, 4])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample005.svg'), transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=7,
                     zoom=True,
                     zoom_row=[0, 2],
                     zoom_col=[0, 2])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample006.svg'), transparent=True)

    obj = matrix(K, ax=ax[1], type='image', 
                      highlight=True, 
                      highlight_row=[0, 1], 
                      highlight_col=[0,1], 
                      highlight_color=hcolor,
                      highlight_width=8,
                     zoom=True,
                     zoom_row=[0, 1],
                     zoom_col=[0, 1])
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample007.svg'), transparent=True)

    obj = matrix(K[:2, :2], ax=ax[1], type='values')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample008.svg'), transparent=True)

    ind = [0, 1]
    obj = matrix(K[ind, ind], ax=ax[1], type='values')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample008.svg'), transparent=True)

    ax[0].cla()
    two_point_pred(K, f.T, x, ax=ax[0],ind=[0, 1], stub='two_point_sample', start=9, diagrams=diagrams)

    ind = [0, 4]
    ax[0]
    obj = matrix(K[ind, ind], ax=ax[1], type='values')
    ax[1].set_xlabel('$i$',fontsize=16)
    ax[1].set_ylabel('$i^\prime$',fontsize=16)
    mlai.write_figure(os.path.join(diagrams, 'two_point_sample013.svg'), transparent=True)

    ax[0].cla()
    two_point_pred(K, f.T, x, ax=ax[0],ind=ind, stub='two_point_sample', start=14, diagrams=diagrams)


def poisson(diagrams='../diagrams'):
    from scipy.stats import poisson
    fig, ax = plt.subplots(figsize=two_figsize)
    y = np.asarray(range(0, 16))
    p1 = poisson.pmf(y, mu=1.)
    p3 = poisson.pmf(y, mu=3.)
    p10 = poisson.pmf(y, mu=10.)

    ax.plot(y, p1, 'r.-', markersize=20, label='$\lambda=1$', lw=3)
    ax.plot(y, p3, 'g.-', markersize=20, label='$\lambda=3$', lw=3)
    ax.plot(y, p10, 'b.-', markersize=20, label='$\lambda=10$', lw=3)
    ax.set_title('Poisson Distribution', fontsize=20)
    ax.set_xlabel('$y_i$', fontsize=20)
    ax.set_ylabel('$p(y_i)$', fontsize=20)
    ax.legend(fontsize=20)
    mlai.write_figure(os.path.join(diagrams, 'poisson.svg'), transparent=True)

def logistic(diagrams='../diagrams'):
    fig, ax = plt.subplots(figsize=two_figsize)
    f = np.linspace(-8, 8, 100)
    g = 1/(1+np.exp(-f))
    
    ax.plot(f, g, 'r-', lw=3)
    ax.set_title('Logistic Function', fontsize=20)
    ax.set_xlabel('$f_i$', fontsize=20)
    ax.set_ylabel('$g_i$', fontsize=20)
    mlai.write_figure(os.path.join(diagrams, 'logistic.svg'), transparent=True)


def height(ax, h, ph):
    "Plot height as a distribution."
    ax.plot(h, ph, '-', color=[1, 0, 0], linewidth=3)
    ax.set_xticks([1.25, 1.7, 2.15])
    ax.set_yticks([1, 2, 3])
    ax.set_xlabel('$h/m$', fontsize=20)
    ax.set_ylabel('$p(h)$', fontsize=20)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.vlines(xlim[0], ylim[0], ylim[1], color='k')
    ax.hlines(ylim[0], xlim[0], xlim[1], color='k')

def weight(ax, w, pw):
    "Plot weight as a distribution."
    ax.plot(w, pw, '-', color=[0, 0, 1.], linewidth=3)
    ax.set_xticks([55, 75, 95])
    ax.set_yticks([0.02, 0.04, 0.06])
    ax.set_xlabel('$w/kg$', fontsize=20)
    ax.set_ylabel('$p(w)$', fontsize=20)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.vlines(xlim[0], ylim[0], ylim[1], color='k')
    ax.hlines(ylim[0], xlim[0], xlim[1], color='k')

def kronecker_illustrate(fontsize=25, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 4, figsize=two_figsize)
    A = [['$a$', '$b$'],
         [ '$c$', '$d$']]
    B = [['$\mathbf{K}$']]

    AkroneckerB = [['$a\mathbf{K}$', '$b\mathbf{K}$'],
                    ['$c\mathbf{K}$', '$d\mathbf{K}$']]
    ax[0].set_position([0, 0, 1, 1])
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[0].text(0.4, 0.5, ' $\otimes$', horizontalalignment='center',
                  fontsize=fontsize)
    ax[0].text(0.55, 0.5, ' $=$', horizontalalignment='center',
                  fontsize=fontsize)

    ax[1].set_position([0.15, 0.4, 0.2, 0.2])
    objA = matrix(A, ax=ax[1], bracket_style='square', type='entries',
                  fontsize=fontsize)


    ax[2].set_position([0.45, 0.45, 0.05, 0.1])
    objB = matrix(B, ax=ax[2], bracket_style='none', type='entries',
                  fontsize=fontsize)
    
    ax[3].set_position([0.57, 0.35, 0.35, 0.3])
    objAkB = matrix(AkroneckerB, ax=ax[3], bracket_style='square', type='entries',
                  fontsize=fontsize)
    ax[0].set_axis_off()
        
    mlai.write_figure(os.path.join(diagrams, 'kronecker_product.svg'), transparent=True)
def blank_canvas(ax):
    """Turn an axis into a blank canvas"""
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_axis_off()
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def kronecker_illustrate(fontsize=25, figsize=two_figsize, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    A = [['$a$', '$b$'],
         [ '$c$', '$d$']]
    B = [['$\mathbf{K}$']]

    AkroneckerB = [['$a\mathbf{K}$', '$b\mathbf{K}$'],
                    ['$c\mathbf{K}$', '$d\mathbf{K}$']]

    blank_canvas(ax[0])
    ax[0].text(0.4, 0.5, ' $\otimes$',
               horizontalalignment='center',
               fontsize=fontsize)
    ax[0].text(0.55, 0.5, ' $=$',
               horizontalalignment='center',
               fontsize=fontsize)

    ax[1].set_position([0.15, 0.4, 0.2, 0.2])
    objA = matrix(A, ax=ax[1], bracket_style='square', type='entries',
                  fontsize=fontsize)


    ax[2].set_position([0.45, 0.45, 0.05, 0.1])
    objB = matrix(B, ax=ax[2], bracket_style='none', type='entries',
                  fontsize=fontsize)
    
    ax[3].set_position([0.57, 0.35, 0.35, 0.3])
    objAkB = matrix(AkroneckerB, ax=ax[3], bracket_style='square', type='entries',
                  fontsize=fontsize)
        
    mlai.write_figure(os.path.join(diagrams, 'kronecker_product.svg'), transparent=True)

def kronecker_IK(fontsize=25, figsize=two_figsize, reverse=False, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    my_rgb = [[1., 1., 1.],[1., 0., 0.],[ 0., 1., 0.],[ 0., 0., 1.]]
    from matplotlib.colors import ListedColormap
    colormap = ListedColormap(my_rgb, name='primary+black')
    dim_I = 3
    dim_K = 3
    I = np.eye(dim_I)
    L = np.tril(np.ones(dim_K))
    K = np.dot(L, L.T)
        
    blank_canvas(ax[0])
    ax[0].text(0.3, 0.5, ' $\otimes$',
               horizontalalignment='center',
               fontsize=fontsize)
    ax[0].text(0.615, 0.5, ' $=$',
               horizontalalignment='center',
               fontsize=fontsize)

    ax[1].set_position([0.05, 0.05, 0.2, 0.9])
    objI = matrix(np.stack([1-I]*3, 2), ax=ax[1],
                  bracket_style='boxes', type='colorpatch',
                  fontsize=fontsize)


    ax[2].set_position([0.35, 0.05, 0.2, 0.9])
    objK = matrix(np.stack((K==1, K==2, K==3), 2),
                  ax=ax[2],
                  bracket_style='boxes', type='colorpatch',
                  fontsize=fontsize)
    if reverse:
        kron_IK = np.kron(K, I)    
    else:
        kron_IK = np.kron(I, K)    
    ax[3].set_position([0.675, 0.1, 0.3, 0.85])
    objAkB = matrix(np.stack((np.logical_or(kron_IK==1, kron_IK==0),
                              np.logical_or(kron_IK==2, kron_IK==0),
                              np.logical_or(kron_IK==3, kron_IK==0)), 2),
                    ax=ax[3],
                    bracket_style='boxes', type='colorpatch',
                    fontsize=fontsize)
    if reverse:
        mlai.write_figure(os.path.join(diagrams, 'kronecker_KI.svg'), transparent=True)
    else:
        mlai.write_figure(os.path.join(diagrams, 'kronecker_IK.svg'), transparent=True)

def kronecker_IK_highlight(fontsize=25, figsize=two_figsize, reverse=False, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    my_rgb = [[1., 1., 1.],[1., 0., 0.],[ 0., 1., 0.],[ 0., 0., 1.]]
    from matplotlib.colors import ListedColormap
    colormap = ListedColormap(my_rgb, name='primary+black')
    dim_I = 3
    dim_K = 3
    I = np.eye(dim_I)
    L = np.tril(np.ones(dim_K))
    K = np.dot(L, L.T)
    if reverse:
        kron_IK = np.kron(K, I)
        stem = 'KI'
    else:
        kron_IK = np.kron(I, K)
        stem = 'IK'
        
    IK_stack = np.stack((np.logical_or(kron_IK==1, kron_IK==0),
                         np.logical_or(kron_IK==2, kron_IK==0),
                         np.logical_or(kron_IK==3, kron_IK==0)), 2)        
    ax.set_position([0, 0, 1, 1])
    objAkB = matrix(IK_stack,
                    ax=ax,
                    bracket_style='boxes', type='colorpatch',
                    fontsize=fontsize)
        
    mlai.write_figure(os.path.join(diagrams, 'kronecker_{stem}_highlighted001.svg').format(stem=stem))
    objAkB = matrix(IK_stack,
                    ax=ax,
                    bracket_style='boxes', type='colorpatch',
                    fontsize=fontsize,
                    highlight=True, 
                    highlight_row=[0, 2], 
                    highlight_col=[0, 2], 
                    highlight_color=hcolor,
                    highlight_width=8)
    mlai.write_figure(os.path.join(diagrams, 'kronecker_{stem}_highlighted002.svg').format(stem=stem))
    count = 2
    for zoom in [6, 3, 2]:
        objAkB = matrix(IK_stack,
                        ax=ax,
                        bracket_style='boxes', type='colorpatch',
                        fontsize=fontsize,
                        highlight=True, 
                        highlight_row=[0, 2], 
                        highlight_col=[0, 2], 
                        highlight_color=hcolor,
                        highlight_width=8,
                        zoom=True,
                        zoom_row=[0, zoom],
                        zoom_col=[0, zoom])
        count+=1
        mlai.write_figure(os.path.join(diagrams, 'kronecker_{stem}_highlighted{count:0>3}.svg').format(stem=stem, count=count))

def kronecker_WX(fontsize=25, figsize=two_figsize, diagrams='../diagrams'):
    """Illustrate a Kronecker product"""
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    A = [['$\mathbf{W}$', '$\mathbf{0}$', '$\mathbf{0}$'],['$\mathbf{0}$', '$\mathbf{W}$', '$\mathbf{0}$'],['$\mathbf{0}$', '$\mathbf{0}$', '$\mathbf{W}$']]
    B = [['$\mathbf{x}_{1,:}$'],['$\mathbf{x}_{2,:}$'],['$\mathbf{x}_{3,:}$']]
    AkroneckerB = [['$\mathbf{W}\mathbf{x}_{1,:}$'],[ '$\mathbf{W}\mathbf{x}_{2,:}$'], ['$\mathbf{W}\mathbf{x}_{3,:}$']]

    blank_canvas(ax[0])
    ax[0].text(0.4, 0.5, r'$\times$',
               horizontalalignment='center',
               fontsize=fontsize)
    ax[0].text(0.65, 0.5, ' $=$',
               horizontalalignment='center',
               fontsize=fontsize)

    ax[1].set_position([0.05, 0.35, 0.3, 0.3])
    objA = matrix(A, ax=ax[1], bracket_style='square', type='entries',
                  fontsize=fontsize)


    ax[2].set_position([0.4, 0.35, 0.25, 0.3])
    objB = matrix(B, ax=ax[2], bracket_style='none', type='entries',
                  fontsize=fontsize)
    
    ax[3].set_position([0.6, 0.35, 0.35, 0.3])
    objAkB = matrix(AkroneckerB, ax=ax[3], bracket_style='square', type='entries',
                  fontsize=fontsize)
        
    mlai.write_figure(os.path.join(diagrams, 'kronecker_WX.svg'), transparent=True)

def perceptron(x_plus, x_minus, learn_rate=0.1, max_iters=10000, max_updates=30, seed=100001, diagrams='../diagrams'):
    w, b, x_select = mlai.init_perceptron(x_plus, x_minus, seed=seed)
    updates = 0
    count = 0
    iterations = 0
    setup=True
    f2, ax2 = plt.subplots(1, 2, figsize=two_figsize)
    handle = init_perceptron(f2, ax2, x_plus, x_minus, w, b)
    handle['plane'].set_visible(False)
    handle['arrow'].set_visible(False)
    handle['circle'] = plt.Circle((x_select[0], x_select[1]), 0.25, color='b', fill=False)
    ax2[0].add_artist(handle['circle'])
    mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.svg').format(samp=count), transparent=True)
    extent = ax2[0].get_window_extent().transformed(f2.dpi_scale_trans.inverted())
    mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.png').format(samp=count), bbox_inches=extent, transparent=True)
    count += 1
    handle['plane'].set_visible(True)
    handle['arrow'].set_visible(True)
    mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.svg').format(samp=count), transparent=True)
    mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.png').format(samp=count), bbox_inches=extent, transparent=True)

    while updates<max_updates and iterations<max_iters:
        iterations += 1
        w, b, x_select, updated = mlai.update_perceptron(w, b, x_plus, x_minus, learn_rate)
        if updated:
            updates += 1
            count+=1
            handle['circle'].center = x_select[0], x_select[1]
            mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.svg').format(samp=count), transparent=True)     
            mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.png').format(samp=count), bbox_inches=extent, transparent=True)        
            count+=1
            handle = update_perceptron(handle, f2, ax2, x_plus, x_minus, updates, w, b)
            mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.svg').format(samp=count), transparent=True)
            mlai.write_figure(figure=f2, filename=os.path.join(diagrams, 'perceptron{samp:0>3}.png').format(samp=count), bbox_inches=extent, transparent=True)
    print('Data passes:', iterations)
    return count

import GPy
def contour_data(data, length_scales, log_SNRs, kernel_call=GPy.kern.RBF):
    """
    Evaluate the GP objective function for a given data set for a range of
    signal to noise ratios and a range of lengthscales.

    :data_set: A data set from the utils.datasets director.
    :length_scales: a list of length scales to explore for the contour plot.
    :log_SNRs: a list of base 10 logarithm signal to noise ratios to explore for the contour plot.
    :kernel: a kernel to use for the 'signal' portion of the data.
    """

    lls = []
    total_var = np.var(data['Y'])
    kernel = kernel_call(1, variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(data['X'], data['Y'], kernel=kernel)
    for log_SNR in log_SNRs:
        SNR = 10.**log_SNR
        noise_var = total_var / (1. + SNR)
        signal_var = total_var - noise_var
        model.kern['.*variance'] = signal_var
        model.likelihood.variance = noise_var
        length_scale_lls = []

        for length_scale in length_scales:
            model['.*lengthscale'] = length_scale
            length_scale_lls.append(model.log_likelihood())

        lls.append(length_scale_lls)

    return np.array(lls)
