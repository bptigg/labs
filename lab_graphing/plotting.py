import matplotlib.pyplot as plt
import matplotlib
from sheet import dataframe
from sheet import fit
from sheet import fitting_types
import numpy as np

axis = None
fitting_curves = []
custom_legend_flag = False

def plotting_menu(sheet: dataframe):
    exit = False
    while(exit == False):
        valid = False
        option = 0
        while(valid == False):
            print("1) Plot data")
            print("2) Add error bars")
            print("3) Plot data fit")
            print("4) Display graph")
            print("5) Add titles")
            print("6) Custom Legend")
            print("7) Exit")
            option = input("Select option (1-7): ")
            try:
                option = int(option)
            except:
                print("Invalid input")
                continue
            if(option < 7 or option >= 1):
                valid = True
        if(option == 1):
            plot_data(sheet)
        elif(option == 2):
            add_error_bars(sheet)
        elif(option == 3):
            plot_functions(sheet)
        elif(option == 4):
            plot_figure()
        elif(option == 5):
            add_titles()
        elif(option == 6):
            custom_legend()
        elif(option == 7):
            print("Returning to previous menu")
            exit = True
            continue
        else:
            print("Invalid input")
            continue

def custom_legend():
    global axis
    global custom_legend_flag
    custom_legend_flag = True
    
    #custom body
    global fitting_curves
    legend = []
    for i in range (0,1):
        legend.append('mu = {} cm^-1'.format(fitting_curves[i].print_function()))
    legend = axis.legend(legend, loc = 0)
    legend.get_frame().set_alpha(0.5)
    

def plot_figure():
    global axis
    if(custom_legend_flag == False):
        legend = axis.legend()
        legend.get_frame().set_alpha(0.5)
    plt.rcParams['font.size'] = '20'
    plt.show(block = False)
    axis = None
    print("Graphing enviroment reset")

def add_titles():
    global axis
    y_title = input("Title for the y axis: ")
    x_title = input("Title for the x axis: ")

    axis.set_xlabel(x_title)
    axis.set_ylabel(y_title)

def initlize_figure():
    figure = plt.figure(figsize = (12, 9), dpi = 100)
    global axis
    axis = figure.add_subplot(111)

def plot_data(data : dataframe, colour = ' '):
    global axis
    if(axis == None):
        initlize_figure()
    if(colour == ' '):
        axis.scatter(data.X, data.Y, color = 'black')
    else:
        axis.scatter(data.X, data.Y, color = colour)

def add_error_bars(data : dataframe, colour = ' '):
    global axis
    if(axis == None):
        initlize_figure()
    if(colour == ' '):
        axis.errorbar(data.X, data.Y, xerr = data.X_U, yerr = data.Y_U, barsabove = True, fmt = 'o', capsize = 2, ecolor = 'g')
    else:
        axis.errorbar(data.X, data.Y, xerr = data.X_U, yerr = data.Y_U, barsabove = True, fmt = 'o', capsize = 2, ecolor = 'g')

def plot_functions(sheet : dataframe):
    for function in sheet.data_fit:
        function.print_function()
        print("Plot function")
        option = 0
        valid = False
        while(valid == False):
            option = input("Yes(1) / No(0): ")
            try:
                option = int(option)
            except:
                print("Invalid input")
                continue
            if(option == 1 or option == 0):
                valid = True
            else:
                print("Input out of range")
                continue
        if(option == 1):
            plot_fitted_function(function)
        

def plot_fitted_function(fit : fit, colour = ' '):
    global axis
    global fitting_types
    if(axis == None):
        initlize_figure()
    model_y = []
    fitting_curves.append(fit)
    x_coord = fit.x
    x_coord.sort()
    x_data = x_coord
    if(len(fit.x) < 50):
        range = x_coord[len(x_coord)-1] - x_coord[0]
        step = range / 10000
        x_data = np.arange(x_coord[0], x_coord[len(x_coord)-1], step)
    for x in x_data:
        value = 0
        if(fit.type == fitting_types.POLYNOMIAL):
            value = fit.curve(float(x), fit.param)
        else:
            value = fit.curve(float(x), *fit.param)
        model_y.append(value)
    if(colour == ' '):
        axis.plot(x_data, model_y, label = fit.type)
    else:
        axis.plot(x_data, model_y, color = colour, label = fit.type)

def plot_measurments(df):
    y = 9
    x = 12

    figure = plt.figure(figsize = (x,y), dpi = 50)
    axis_simple = figure.add_subplot(111)
    axis_simple.scatter(df.X, df.Y, color = 'black')

    plt.show(block = False)
    #plt.close()

def plot_line_measurments(df):
    y = 9
    x = 12

    figure = plt.figure(figsize = (x,y), dpi = 50)
    axis_simple = figure.add_subplot(111)
    axis_simple.plot(df.X, df.Y, color = 'black')

    plt.show(block = False)

def plot_fit(fit : fit):
    figure = plt.figure(figsize = (12,9), dpi = 50)
    axis_simple = figure.add_subplot(111)
    axis_simple.scatter(fit.x, fit.y, color = 'black')
    model_y = []
    x_coord = fit.x
    x_coord.sort()
    x_data = x_coord
    if(len(fit.x) < 50):
        range = x_coord[len(x_coord)-1] - x_coord[0]
        step = range / 10000
        x_data = np.arange(x_coord[0], x_coord[len(x_coord)-1], step)
    for x in x_data:
        value = 0
        if(fit.type == fitting_types.POLYNOMIAL):
            value = fit.curve(float(x), fit.param)
        else:
            value = fit.curve(float(x), *fit.param)
        model_y.append(value)
    axis_simple.plot(x_data, model_y, color = 'blue')

    plt.show(block = False)

def plot_just_fit(fit : fit):
    figure = plt.figure(figsize = (12,9), dpi = 50)
    axis_simple = figure.add_subplot(111)
    model_y = []
    x_coord = fit.x
    x_coord.sort()
    x_data = x_coord
    if(len(fit.x) < 50):
        range = x_coord[len(x_coord)-1] - x_coord[0]
        step = range / 10000
        x_data = np.arange(x_coord[0], x_coord[len(x_coord)-1], step)
    for x in x_data:
        value = 0
        if(fit.type == fitting_types.POLYNOMIAL):
            value = fit.curve(float(x), fit.param)
        else:
            value = fit.curve(float(x), *fit.param)
        model_y.append(value)
    axis_simple.plot(x_data, model_y, color = 'blue')

    plt.show(block = False)
