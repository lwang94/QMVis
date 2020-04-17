# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:55:49 2019

@author: lawre
"""
from flatten import flatten

import matplotlib.animation as animation

class PlotLevels:
    """
    Create a set of axes in a figure such that there are multiple sub-axes stacked vertically on top of each other (by defaults, can 
    customize coordinates when initializing instance) with a single axis in the background that can be used to plot information
    universal to all sub-axes eg. plotting n-eigenfunctions of a potential with the potential plotted in the background.
    
    Parameters
    -----------------------------------------------------
    fig : matplotlib.figure.Figure
        The figure object that is used to contain the axes
    nlevels : `int`
        The number of sub-axes to be created in the foreground.    
    coordinates : `list`, optional
        The coordinates of each subaxes. If left empty, the coordinates will be such that each sub-axes in the foreground is uniform 
        in size and stacked vertically on top of each other. Defaults to []
    show_axes : `boolean`
        Boolean determining whether the figure will show axes ticks and labels. Defaults to False.
        
    Attributes
    -------------------------------------------------------
    fig : matplotlib.figure.Figure
        Same as in Parameters
    axes : list of matplotlib.axes.Axes
        A list of size, nlevels + 1, containing the axes objects present in the figure. The first item in the list will always be the
        single background axis.
    
    DOUBLE CHECK FORMULA FOR HARMONIC OSCILLATOR BSP (AND ALL OTHER BSP's FOR THAT MATTER) IS CORRECT
    """
    
    def __init__(self, fig, nlevels, coordinates = [], show_axes = False):
        self.fig = fig
        
        #Create the set of sub-axes and background axis in the figure
        #Default option
        if coordinates == []:        
            self.axes = [self.fig.add_axes([0.1, 0.05, 0.8, 0.85])]
            for i in range(nlevels):
                self.axes += [self.fig.add_axes([0.1, 
                                                 0.1 + i * 0.8 / nlevels, 
                                                 0.8, 
                                                 0.8 / nlevels], 
                                                sharex = self.axes[0])] #uniform spacing   
                self.axes[-1].patch.set_alpha(0)
                
        #When the user want to use custom coordinates but the number of coordinates don't match the number of axes needed
        elif len(coordinates) != nlevels + 1:
            raise Exception(f'Number of sets of coordinates ({len(coordinates)}) must match number of levels + 1 ({nlevels + 1})')
            
        #When the user wants to use custom coordinates and inputs it correctly
        else:
            self.axes = [self.fig.add_axes(coordinates[0])]
            for i in range(nlevels):
                self.axes += [self.fig.add_axes(coordinates[i+1], sharex = self.axes[0])]
                self.axes[-1].patch.set_alpha(0)

        #Hide the axes ticks and labels to remove clutter
        if show_axes == False:
            for i in range(0, len(self.axes)):
                self.axes[i].set_axis_off()
    
    
    def plot_multipleaxes(self, x, y_data, **kwargs):
        """
        Plots line graphs on multiple axes.
        
        Parameters
        ----------------------------------------------
        x : array-like
            The x-data
        y_data : `dict`
            Dictionary containing the y-data. 
            
            keys --> `int`, the index of the axis on which you wish to plot the corresponding y-data
            
            values --> `array-like`, the corresponding y-data
        data_color : `dict`, optional
            Dictionary containing the colors of the line representing the data.
            
            keys --> `int`, the index of the axis on which you wish to change the color of the data.
            
            values --> `str`, Accepts characters compatible with the matplotlib.lines.Line2D.set_color() method. See 
            (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_color) for more details.
        data_style : `dict`, optional
            Dictionary containing the linestyles of the line representing the data.
            
            keys --> `int`, the index of the axis on which you wish to change the linestyle of the data
            
            values -->  `char`, Accepts characters compatible with the matplotlib.lines.Line2D.set_linestyle() method. See 
            (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle) for more details
        levels_line : `dict`, optional
            Dictionary containing the format strings to determine the color and style of the line showing x-axis (ie. y = 0) for each 
            axis. If not specified, the line will not be shown.
            
            keys --> `int`, the index of the axis on which you wish to show the y = 0 line.
            
            values --> `char`, Accepts characters compatible with matplotlib.pyplot.plot(). See 
            (https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html) for more details.
        levels_label : `dict`, optional
            Dictionary containing tuples that specify the contents, size, and coordinates of labels for each axis. If not specified,
            the axis will not be labelled.
            
            key --> `int`, the index of the axis on which you wish to label
            
            values --> `tuple`, each tuple must contain four item where:
            item1 is the contents of label (`str`), item2 is the font size (`int`), item3 is the x-coordinate of label (`float`), 
            and item4 is the y-coordinate of label (`float`)
        
        Notes
        -----------------------------------------------------
        If the user wishes to customize the plots further, it may be more beneficial to plot the y-data separately by using the matplotlib
        plot method on each axis within the instance and utilizing the full customization capabilities of the matplotlib library.
        Reminder: to call a specific axis within the instance, simply use PlotLevels.axes[`int`, the index of the axis].
        """
        #Plot the data
        lines = {}
        for i in y_data:
            line, = self.axes[i].plot(x, y_data[i])
            lines = {**lines, i : line} #put plots in dictionary for customization later
        
        #Customize the plot based on keyword arguments
        if kwargs.get('data_color') != None:
            for i in kwargs['data_color']:
                lines[i].set_color(kwargs['data_color'][i])
        if kwargs.get('data_style') != None:
            for i in kwargs['data_style']:
                lines[i].set_linestyle(kwargs['data_style'][i])                
        if kwargs.get('levels_line') != None:
            for i in kwargs['levels_line']:
                self.axes[i].plot(x, [0]*len(x), kwargs['levels_line'][i])
        if kwargs.get('levels_label') != None:
            for i in kwargs['levels_label']:
                level_label = kwargs['levels_label'][i]
                self.axes[i].text(level_label[2], # x-coordinate
                                  level_label[3], # y-coordinate
                                  level_label[0], # string
                                  transform = self.axes[i].transAxes, 
                                  fontsize = level_label[1]) # font-size
                                                     
    
    def plot_multipleanimated(self, animation_dict, interval = 200, blitting = True):
        """
        Plots animated line graph on multiple axes. 
        
        Parameters
        --------------------------------------------------
        animation_dict : `dict`
            Dictionary containing instructions for creating multiple animated line graphs.
            
            key --> `int`, the index of the axis on which you wish to plot the animation
            
            value --> `dict`, Dictionary containing the instructions Below are the instructions for constructing the dictionary. The 
            method PlotLevels.create_animation_dict() can be used to more easily construct this parameter.
            
            +--------------+-----------+----------------------------------------------------------------------------------+
            |Key           |Value Type |Description                                                                       |
            +==============+===========+==================================================================================+
            |plot_elements |`dict`     |Dictionary containing the elements to be included in the graph. The keys are:     |
            |              |           |                                                                                  |
            |              |           |``"animatedline"``                                                                |
            |              |           |Boolean determining whether an animated line will be plotted (`bool`)             |
            |              |           |                                                                                  |
            |              |           |``"tracker"``                                                                     |
            |              |           |Boolean determining whether to keep track of the dependance (`bool`)              |
            +--------------+-----------+----------------------------------------------------------------------------------+
            |func          |`callable` |The animated function to be plotted on the axis. Must be acceptable               |
            |              |           |by matplotlib.animation.FuncAnimation(). See                                      | 
            |              |           |https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.animation.FuncAnimation.html  |
            |              |           |for more details.                                                                 |
            +--------------+-----------+----------------------------------------------------------------------------------+   
            |arguments     |`tuple`    |Tuple containing the arguments for the above func obejct. Does not include the    |
            |              |           |frame and plot_elements parameter (typically i and plot_elements)                 |
            +--------------+-----------+----------------------------------------------------------------------------------+
            
        interval : `int`, optional
            The interval in milliseconds between each frame of the animation. Defaults to 200
        blitting : `bool`, optional
            Boolean determining whether the animation function will use blitting. Set to False when creating animations on both 
            the background and foreground axes. Defaults to True.
        
        Notes
        -------------------------------------------------------------------
        The method PlotLevels.create_animation_dict() can be used to more easily construct the animation_dict parameter.
        """
        #create separate dictionary for plot elements
        plot_elements = {}
        for axis in animation_dict:
            plot_elements[axis] = {'animatedline': None,
                                   'tracker': None}
        
        #insert matplotlib artist objects into the plot_elements dictionary based on the animation_dict parameter
        for axis in animation_dict:
            #create Line2D artist object
            if animation_dict[axis]['plot_elements'].get('animatedline') == True:
                plot_elements[axis]['animatedline'], = self.axes[axis].plot([], [])
            else:
                plot_elements[axis].pop('animatedline', None) #remove the animatedline key from plot_elements if set to False in animated_dict
            #create Text artist object  
            if animation_dict[axis]['plot_elements'].get('tracker') == True:
                plot_elements[axis]['tracker'] = self.axes[axis].text(0.05, 0.9, '', 
                                                                      transform = self.axes[axis].transAxes, 
                                                                      fontsize = 'x-large')
            else:
                plot_elements[axis].pop('tracker', None) #remove the trakcer key from plot_elements if set to False in animated_dict
                
        #init function for blitting when calling animation.FuncAnimation()
        def init():
            pev = [element for axis in plot_elements for element in flatten(plot_elements[axis].values())] #returns one dimensional list containing the elements the user wants for the plot
            
            #ensure the return line returns an iterable of artists            
            if len(pev) == 1:
                pev = (pev[0],)
            else:
                pev = tuple(pev)
            
            return pev
        
        #separate cases for blitting
        if blitting == True:
            ani = animation.FuncAnimation(self.fig, 
                                          self.animate, 
                                          init_func = init, 
                                          fargs = [plot_elements, animation_dict], 
                                          interval = interval,
                                          blit = True)
        else:
            ani = animation.FuncAnimation(self.fig, 
                                          self.animate, 
                                          fargs = [plot_elements, animation_dict], 
                                          interval = interval,
                                          blit = False)
        
        return ani
    
    
    def animate(self, i, plot_elements, animation_dict):
        """
        Returns iterable of artist objects for use in PlotLevel.plot_multipleanimated().
        
        Parameters
        -------------------------------------------------
        i : number, iterable
            The frame of the animation
        plot_elements : `dict`
            Dictionary containing artist objects.
            
            key --> `int`, the index of the axis on which you wish to plot the animation
            
            values --> `dict`, Dictionary containing artist objects for the function to act on. See below for keys and values.
            
                 +------------+-----------------------+
                 |Keys        |Value                  |
                 +============+=======================+
                 |animatedline|matplotlib.line.Line2D |
                 +------------+-----------------------+
                 |tracker     |matplotlib.text.Text   |
                 +------------+-----------------------+
        animation_dict : `dict`
            Same as animation_dict in PlotLevels.plot_multipleanimated but without the plot_elements keyword. See
            PlotLevels.plot_multipleanimated for more details
        """
        #create tuple of artist objects after they've gone through the functions defined in animation_dict[axis]['func']
        pev = ()
        for axis in animation_dict:
            pev += animation_dict[axis]['func'](i, plot_elements[axis], *animation_dict[axis]['arguments'])
        
        #ensures function returns iterable even if only one artist object is returned
        if len(pev) == 1:
            pev = (pev[0],)
            
        return pev
    
    
    def create_animation_dict(self, axes, plot_elements, func, arguments):
        """
        Creates and returns a dictionary that can be passed into the PlotLevels.plot_multipleanimated() method as the animation_dict 
        parameter.
        
        Parameters
        -----------------------------------------------------
        axes : list of int
            List containing the indices of the axis to be plotted
        plot_elements : list of dict
            List containing the dictionaries which determine what elements will be plotted on which axis. The keys and values for
            each dictionary are below:
        
            ``"animatedline"``
            Boolean determining whether an animated line will be plotted (`bool`). 
            
            ``"tracker"``
            Boolean determining whether to keep track of the dependance (`bool`). 
        func : list of callables
            List containing the animation function to be used on each axis. Must be acceptable by matplotlib.animation.FuncAnimation().
            See https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.animation.FuncAnimation.html for more details.
        arguments : list of tuples
            List of tuples which contain the arguments to be passed into the above callable functions. This does not include the frame
            and plot_elements parameter (typically i and plot_elements)
        
        Returns
        -------------------------------------------------------
        out : `dict`
            Dictionary in the proper format to be passed into the animation_dict parameter of PlotLevels.plot_multipleanimated() method.
            
        Examples
        --------------------------------------------------------
        Return the animation_dict to plot 3 time dependance animations of the superposition of 100 uniformly distributed eigenfunctions
        of an infinite square well of size, L = 2. The 3 animations will appear on 3 subaxes stacked vertically on top of each other.
        
        >>> import QuantumSuperposition as qs
        >>> import BoundStatePotentials as bsp
        >>> import Particle as p
        >>> x= np.linspace(-1, 1, 1000)
        >>> superposition = qs.QuantumSuperposition(bsp.InfSquareWell(x, 2), range(1, 101))
        >>> fig = plt.figure()
        >>> graph = PlotLevels(fig, 3)
        >>> axes = [1, 2, 3]
        >>> plot_elements = [{'animatedline' : True,
        ...                   'tracker' : True}]*3
        >>> func = [superposition.animate_time]*3
        >>> arguments = [([0, 2], p.Particle(1), 50, 1, [], {})]*3 #particle of mass = 1 and h_bar = 1
        >>> graph.create_animation_dict(axes, plot_elements, func, arguments)
        {1: {'plot_elements': {'animatedline': True, 'tracker': True}, 
             'func': <bound method QuantumSuperposition.animate_time of <QuantumSuperposition.QuantumSuperposition object at 0x00000199831DE940>>, 
             'arguments': ([0, 2], <Particle.Particle object at 0x00000199FDCB1710>, 50, 1, [], {})}, 
         2: {'plot_elements': {'animatedline': True, 'tracker': True}, 
             'func': <bound method QuantumSuperposition.animate_time of <QuantumSuperposition.QuantumSuperposition object at 0x00000199831DE940>>, 
             'arguments': ([0, 2], <Particle.Particle object at 0x00000199FDCB1710>, 50, 1, [], {})}, 
         3: {'plot_elements': {'animatedline': True, 'tracker': True}, 
             'func': <bound method QuantumSuperposition.animate_time of <QuantumSuperposition.QuantumSuperposition object at 0x00000199831DE940>>, 
             'arguments': ([0, 2], <Particle.Particle object at 0x00000199FDCB1710>, 50, 1, [], {})}}
        """
        animation_dict = {axes[i] : {'plot_elements' : plot_elements[i], 
                                     'func' : func[i], 
                                     'arguments' : arguments[i]} for i in range(len(axes))}        
        return animation_dict
            
            
        