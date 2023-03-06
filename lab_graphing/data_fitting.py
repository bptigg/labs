import numpy as np
import math
from polynomial import Polynomial
from polynomial import evaluate
from scipy import optimize as spo
from sheet import fit
from sheet import fitting_types
from sheet import dataframe
import plotting
import sys


max_steps = 999
tolerance = 1e-7

def diffrentiate_polynomail(function : fit) -> fit:
    new_param = []
    df = fit(fitting_types.NULL, [], [], [], None, 0, [])
    for i in range(0,len(function.param)-1):
        new_param.append(function.param[i] * (len(function.param) - 1 - i))
    df.set_to(fit(fitting_types.POLYNOMIAL, function.x, function.y, [], function.curve, len(new_param), new_param))
    return df

def guassian_peak_fit(model : fit) -> fit:
    sys.setrecursionlimit(1000)
    return_fit = fit(fitting_types.GUASSIAN, [], [], [], None, 3, [])
    if(model.type != fitting_types.POLYNOMIAL and model.type != fitting_types.COSINE):
        print("Invalid model fit")
        return return_fit
    f = fit(fitting_types.NULL, [], [], [], None, 0, [])
    df = fit(fitting_types.NULL, [], [], [], None, 0, [])
    if(model.type == fitting_types.POLYNOMIAL):
        f.set_to(diffrentiate_polynomail(diffrentiate_polynomail(model)))
        df.set_to(diffrentiate_polynomail(f))
    else:
        f_curve = lambda x, b, c, d, e, f, g: -(b*c*c)*np.sin(c*x + d) - (e*f*f)*np.cos(f*x + g)
        new_param = model.param[1:len(model.param)]
        f.set_to(fit(fitting_types.COSINE, model.x, model.y, [], f_curve, len(new_param), new_param))
        df_curve = lambda x, b, c, d, e, f, g: -(b*c*c*c)*np.cos(c*x + d) + (e*f*f*f)*np.sin(f*x + g)
        df.set_to(fit(fitting_types.COSINE, model.x, model.y, [], df_curve, len(new_param), new_param))
        
    
    plotting.plot_just_fit(f)
    plotting.plot_just_fit(df)

    avg_spacing = (model.x[-1] - model.x[0]) / len(model.x)
    step = avg_spacing / 100
    
    valid = False
    estimate = 0
    while(valid == False):
        estimate = input("Estimate of peak x value: ")
        try:
            estimate = float(estimate)
        except:
            print("Invalid input")
            continue
        valid = True
    
    peak_loc = find_peak(estimate, model)

    print("Peak location at {}".format(peak_loc))

    max = peak_loc
    iteration = 0
    x_0 = give_estimate_of_x_o()
    while max <= peak_loc:
        iteration = iteration + 1
        if(x_0 == None or iteration != 1):
            x_0 = peak_loc + (step*iteration)
        max = Newton_Raphson_Method(f, df, x_0, 0)
        if(max == None):
            max = peak_loc
        if(iteration > 10000):
            max = model.x[len(model.x)-1]
            break

    min = peak_loc
    iteration = 0
    x_0 = give_estimate_of_x_o()
    while min >= peak_loc:
        iteration = iteration + 1
        if(x_0 == None or iteration != 1):
            x_0 = peak_loc - (step*iteration)
        min = Newton_Raphson_Method(f, df, x_0, 0)
        if(min == None):
            min = peak_loc
        if(iteration > 10000):
            min = model.x[0]
            break
    
    print("Peak begins at {}".format(min))
    print("Peak ends at {}".format(max))

    actual_min = 0
    possible_min = []
    for i in model.x:
        if i <= min:
            if((i + avg_spacing) - min) >= 0:
                possible_min.append(i)
        else:
            if((i - avg_spacing) - min) <= 0:
                possible_min.append(i)

    diff = []
    for x in possible_min:
        diff.append(abs(min - x))

    diff.sort()
    min_value = diff[0]
    index = diff.index(min_value)
    actual_min = possible_min[index]

    actual_max = 0
    possible_max = []
    for i in model.x:
        if i <= max:
            if((i + avg_spacing) - max) >= 0:
                possible_max.append(i)
        else:
            if((i - avg_spacing) - max) <= 0:
                possible_max.append(i)

    residuals = []
    for x in possible_max:
        residuals.append(abs(max-x))
    
    residuals.sort()
    min_value = residuals[0]
    index = residuals.index(min_value) 
    actual_max = possible_max[index]

    min_index = model.x.index(actual_min)
    max_index = model.x.index(actual_max)
    sample_x = model.x[min_index : max_index]
    sample_y = model.y[min_index : max_index]

    mean = sum(sample_x) / len(sample_x)
    sd = 0
    temp = 0
    for i in sample_x:
        temp = temp + math.pow(i - mean,2)
    temp = temp / (len(sample_x) - 1)
    sd = math.sqrt(temp)

    print("Mean = {}".format(mean))
    print("Standard deviation = {}".format(sd))

    gaussian_fit = gaussian(sample_x, sample_y)
    plotting.plot_fit(gaussian_fit)

    print("Fitted Gaussian")
    print("Mean = {}".format(gaussian_fit.param[2]))
    print("Standard deviation = {}".format(gaussian_fit.param[1]))
    print("Offset = {}".format(gaussian_fit.param[0]))

    return gaussian_fit

def gaussian_fit_sampe(data : dataframe):
    min = float(input("Start of sample: "))
    max = float(input("End of sample: "))

    avg_spacing = (data.X[-1] - data.X[0]) / len(data.X)

    actual_min = 0
    possible_min = []
    for i in data.X:
        if i <= min:
            if((i + avg_spacing) - min) >= 0:
                possible_min.append(i)
        else:
            if((i - avg_spacing) - min) <= 0:
                possible_min.append(i)

    diff = []
    for x in possible_min:
        diff.append(abs(min - x))

    diff.sort()
    min_value = diff[0]
    index = diff.index(min_value)
    actual_min = possible_min[index]

    actual_max = 0
    possible_max = []
    for i in data.X:
        if i <= max:
            if((i + avg_spacing) - max) >= 0:
                possible_max.append(i)
        else:
            if((i - avg_spacing) - max) <= 0:
                possible_max.append(i)

    residuals = []
    for x in possible_max:
        residuals.append(abs(max-x))
    
    residuals.sort()
    min_value = residuals[0]
    index = residuals.index(min_value) 
    actual_max = possible_max[index]

    min_index = data.X.index(actual_min)
    max_index = data.X.index(actual_max)
    sample_x = data.X[min_index : max_index]
    sample_y = data.Y[min_index : max_index]

    mean = sum(sample_x) / len(sample_x)
    sd = 0
    temp = 0
    for i in sample_x:
        temp = temp + math.pow(i - mean,2)
    temp = temp / (len(sample_x) - 1)
    sd = math.sqrt(temp)

    print("Mean = {}".format(mean))
    print("Standard deviation = {}".format(sd))

    gaussian_fit = gaussian(sample_x, sample_y)
    plotting.plot_fit(gaussian_fit)

    print("Fitted Gaussian")
    print("Mean = {}".format(gaussian_fit.param[2]))
    print("Standard deviation = {}".format(gaussian_fit.param[1]))
    print("Offset = {}".format(gaussian_fit.param[0]))

    return gaussian_fit

def give_estimate_of_x_o():
    print("Give estimate of X_0")
    option = 0
    valid = False
    while(valid == False):
        option = input("Yes(1) / No(0): ")
        try:
            option = int(option)
        except:
            print("Invalid input")
            continue

        if(option == 0 or option == 1):
            valid = True
        else:
            print("Invalid input")
            continue
    if(option == 1):
        estimate = 0
        valid = False
        while(valid == False):
            estimate = input("Give estimate: ")
            try:
                estimate = float(estimate)
            except:
                print("Invalid input")
                continue
            valid = True
        return estimate
    else:
        return None

def find_peak(estimate : float, model : fit) -> float:
    f = fit(fitting_types.NULL, [], [], [], None, 0, [])
    df = fit(fitting_types.NULL, [], [], [], None, 0, [])
    if(model.type == fitting_types.POLYNOMIAL):
        f.set_to(diffrentiate_polynomail(model))
        df.set_to(diffrentiate_polynomail(f))
    else:
        f_curve = lambda x, b, c, d, e, f, g: (b*c)*np.cos(c*x + d) - (e*f)*np.sin(f*x + g)
        new_param = model.param[1:len(model.param)]
        f.set_to(fit(fitting_types.COSINE, model.x, model.y, [], f_curve, len(new_param), new_param))
        df_curve = lambda x, b, c, d, e, f, g: -(b*c*c)*np.sin(c*x + d) - (e*f*f)*np.cos(f*x + g)
        df.set_to(fit(fitting_types.COSINE, model.x, model.y, [], df_curve, len(new_param), new_param))

    return Newton_Raphson_Method(f, df, estimate,0)


def Newton_Raphson_Method(f : fit, df : fit, x_0 : float, steps : int):
    if(steps > max_steps):
        #print("No soloution found")
        return None
    if f.type == fitting_types.POLYNOMIAL:
        value = f.curve(x_0, f.param)
        df_value = df.curve(x_0, df.param)
    else:
        value = f.curve(x_0, *f.param)
        df_value = df.curve(x_0, *df.param)
    if(abs(value) < tolerance):
        return x_0
    else:
        return Newton_Raphson_Method(f, df, x_0 - (value/df_value), steps+1)


def find_best_fit(dataframe):
    error = []
    fits = []
    functions  = [linear_regression, cosine, exponetial, logarithmic, natural_logarithmic, gaussian]
    for x in functions:
        r_sqaured = x(dataframe.X, dataframe.Y, [])
        value = math.fabs(r_sqaured.r_squared)
        if (math.fabs(value) <= 1):
            error.append(math.fabs(r_sqaured.r_squared))
            fits.append(r_sqaured)
    polynomial_error = []
    poly_fits = []
    max_order = 100
    if(len(dataframe.X) < 100):
        max_order = len(dataframe.X)
    for i in range(2, max_order):
        temp = Polynomial(i, 'x', [])
        poly_fit = polynomial_fit(dataframe.X, dataframe.Y, [], temp)
        value = math.fabs(poly_fit.r_squared)
        if(value <= 1):
            polynomial_error.append(math.fabs(poly_fit.r_squared))
            poly_fits.append(poly_fit)
        #plotting.plot_fit(poly_fit)
    max_r_2 = max(polynomial_error)
    for x in poly_fits:
        if(x.r_squared == max_r_2):
            fits.append(x)
            break
    error.append(max_r_2)
    max_value = max(error)
    index = 0
    for x in error:
        if(x == max_value):
            break
        index = index + 1
    best_fit = fits[index]
    plotting.plot_fit(best_fit)
    print(best_fit.r_squared)
    best_fit.print_function()
    return best_fit

def fit_function_menu(sheet: dataframe):
    exit = False
    plotting.plot_measurments(sheet)
    while(exit == False):
        valid = False
        option = 0
        while(valid == False):
            print("1) Fit the best function")
            print("2) Linear fit")
            print("3) Polynomial fit")
            print("4) Sinusoidal fit")
            print("5) Exponetial fit")
            print("6) Natural logarithmic fit")
            print("7) Lograthmic fit")
            print("8) Gaussian fit")
            print("9) Custom fit")
            print("10) Exit ")
            option = input("Select option (1-10): ")
            try:
                option = int(option)
            except:
                print("Invalid input")
                continue
            if(option < 11 or option >= 1):
                valid = True
        
        x =     sheet.X
        y =     sheet.Y
        x_u =   sheet.X_U
        y_u =   sheet.Y_U
        
        fit = None
        second_fit = None
        if(option == 1):
            find_best_fit(sheet)
        elif(option == 2):
            fit = linear_regression(x, y, y_u)
        elif(option == 3):
            valid = False
            order = None
            while(valid == False):
                order = input("Order of the polynomial, e.g x^2 + 2x + 6 is a 2nd order polynomial: ")
                try:
                    order = int(order)
                except:
                    print("Invalid input")
                    continue
                valid = True
            order = order - 1
            temp_polynomial = Polynomial(order, 'x', [])
            fit = polynomial_fit(x, y, y_u, temp_polynomial)
        elif(option == 4):
            fit = cosine(x,y,y_u)
        elif(option == 5):
            fit = exponetial(x,y,y_u)
        elif(option == 6):
            fit = natural_logarithmic(x,y,y_u)
        elif(option == 7):
            fit = logarithmic(x,y,y_u)
        elif(option == 8):
            fit = gaussian(x,y,y_u)
        elif(option == 9):
            fit, second_fit = custom_fit(x,y,y_u)
        elif(option == 10):
            print("Returning to previous menu")
            exit = True
            continue
        else:
            print("Invalid input")
            continue

        if(fit != None):
            plotting.plot_fit(fit)
            print("R^2 value is {}".format(fit.r_squared))
            print("Function is: ")
            fit.print_function()
            valid = False
            option = 0
            while(valid == False):
                print("Keep fitted function")
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
                sheet.add_data_fit(fit)
                print("fitted function added")
            else:
                print("fitted function not added")

        if(second_fit != None):
            plotting.plot_fit(second_fit)
            print("R^2 value is {}".format(second_fit.r_squared))
            print("Function is: ")
            second_fit.print_function()
            valid = False
            option = 0
            while(valid == False):
                print("Keep fitted function")
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
                sheet.add_data_fit(second_fit)
                print("fitted function added")
            else:
                print("fitted function not added")




def linear_regression(x, y, y_u): #add errors
    if((len(x) != len(y)) and (len(y) != len(y_u))):
        print("Data doesn't have the same dimensions, can't compute")
        return

    x_mat = []
    y_mat = []

    for i in range(0,len(x)):
        tempx = []
        tempy = []

        tempx.append(1)
        tempx.append(x[i])

        tempy.append(y[i])

        x_mat.append(tempx)
        y_mat.append(tempy)

    F = np.array(x_mat, dtype = float)
    Y = np.array(y_mat, dtype = float)
    Ft = np.transpose(F)
    FtX = np.dot(Ft, F)
    FtXi = np.linalg.inv(FtX)
    FtXiFt = np.dot(FtXi, Ft)
    coeff = np.dot(FtXiFt, Y)
    #print(coeff)

    linear_fit = fit(fitting_types.LINEAR, x, y, [], lambda x, a,b: a + b*x, 2, coeff)
    r_sqaured(linear_fit, x, y)
    return linear_fit

def polynomial_fit(x,y, y_u, polynomial):
    if((len(x) != len(y)) and (len(y) != len(y_u))):
        print("Data doesn't have the same dimensions, can't compute")
        return
    
    coeff = np.polyfit(x,y,polynomial.order)

    polynomial_fit = fit(fitting_types.POLYNOMIAL, x, y, [], evaluate, len(coeff), coeff)
    r_sqaured(polynomial_fit, x, y)
    return polynomial_fit


def non_linear(fitting_curve, x, y, num_of_param, y_u = []):
    temp_dataframe = dataframe("", [x,y,[],[]])
    plotting.plot_measurments(temp_dataframe)
    p_0 = []
    for i in range(0,num_of_param):
        guess = input("give estimate for parameter {}: ".format(i))
        guess = float(guess)
        p_0.append(guess)
    sigma = []
    for i in range(0, len(y_u)):
        row = []
        for e in range(0, len(y_u)):
            row.append(0)
        sigma.append(row)
        sigma[i][i] = y_u[i]

    if(all_zero(y_u)):
        y_u = []

    coeff = []
    covariance = []
    valid = True
    bounds = int(input("Bounds yes(1) / no(0): "))
    bounds_list = []
    upper = []
    lower = []
    if(bounds == 1):
        for i in range(0,num_of_param):
            print("Parameter {}".format(i))
            upper_bound = float(input("upper bound: "))
            lower_bound = float(input("lower bound: "))
            bounds_list.append([lower_bound, upper_bound])

        for i in bounds_list:
            lower.append(i[0])
            upper.append(i[1])
    bounds_tuple = (lower, upper)
            
    try:
        if(bounds == 1):
            coeff, covariance = spo.curve_fit(fitting_curve, xdata = x, ydata = y, p0 = p_0, bounds = bounds_tuple)
        else:
            coeff, covariance = spo.curve_fit(fitting_curve, xdata = x, ydata = y, p0 = p_0)
    except RuntimeError:
        coeff = p_0
        valid = False
    #print(coeff)
    return coeff, covariance, valid
    
def cosine(x_data, y_data, y_u = []):
    cosine_function = lambda x, a,b,c,d,e,f,g: a + b*np.sin(c * x + d) + e * np.cos(f*x + g) 
    print("a + bsin(cx + d) + ecos(fx + g)")
    return_data = non_linear(cosine_function, x_data, y_data, 7, y_u)
    cosine_fit = fit(fitting_types.COSINE, x_data, y_data , return_data[1], cosine_function, 7, return_data[0])
    r_sqaured(cosine_fit, x_data, y_data)
    return cosine_fit

def exponetial(x, y, y_u = []):
    exponetial_function = lambda x, a, b, c, d: a + b*np.exp(c*x + d)
    print("a + b*exp(cx+d)")
    return_data = non_linear(exponetial_function, x, y, 4, y_u)
    exponetial_fit = fit(fitting_types.EXPONETIAL, x, y, return_data[1], exponetial_function, 4, return_data[0])
    r_sqaured(exponetial_fit, x, y)
    return exponetial_fit

def natural_logarithmic(x,y, y_u = []):
    logarithmic_function = lambda x, a, b, c, d: a + b*np.log(c*x + d)
    print("a + b ln(cx + d)")
    return_data = non_linear(logarithmic_function, x, y, 4, y_u)
    logarithmic_fit = fit(fitting_types.LOGARITHMIC_N, x, y, return_data[1], logarithmic_function, 4, return_data[0])
    r_sqaured(logarithmic_fit, x, y)
    return logarithmic_fit

def logarithmic(x,y,y_u = []):
    logarithmic_function = lambda x, a, b, c, d, e: a + b*(np.log(c*x + d) / np.log(e))
    print("a + b log(cx + d) base e")
    return_data = non_linear(logarithmic_function, x, y, 5, y_u)
    logarithmic_fit = fit(fitting_types.LOGARITHMIC, x, y, return_data[1], logarithmic_function, 5, return_data[0])
    r_sqaured(logarithmic_fit, x, y)
    return logarithmic_fit

def gaussian(x, y, y_u = []):
    gaussian_function = lambda x, a, b, c, d :a + d * np.exp(-0.5 * ((x - c)/b)**2)
    print("Gussian curve")
    return_data = non_linear(gaussian_function, x, y, 4, y_u)
    gaussian_fit = fit(fitting_types.GUASSIAN, x, y, return_data[1], gaussian_function, 4, return_data[0])
    r_sqaured(gaussian_fit, x, y)
    return gaussian_fit

def custom_fit(x,y,y_u = []):
    num_param = int(input("Number of parameters: "))
    param = []
    for i in range(0,num_param):
        param.append(float(input("Parameter {}: ".format(i+1))))
    custom_func = lambda x, a : 162673.9236 * np.exp(-1*a*x)
    if(custom_func == None):
        return fit(fitting_types.NULL, x, y, [], custom_func, 0, [])
    custom_fit = fit(fitting_types.CUSTOM, x, y, [], custom_func, len(param), param)
    return_data = non_linear(custom_func, x, y, 1, y_u)
    exact_fit = fit(fitting_types.CUSTOM, x, y, [], custom_func, len(param), return_data[0])
    r_sqaured(custom_fit, x, y)
    r_sqaured(exact_fit, x, y)
    return custom_fit, exact_fit

def r_sqaured(fitted_curve, x, y):
    r_sqaured_value = 0

    coeff = []
    if(fitted_curve.type == fitting_types.LINEAR):
        for i in range(0,len(fitted_curve.param)):
            coeff.append(fitted_curve.param[i][0])
    else:
        for i in range(0,len(fitted_curve.param)):
            coeff.append(fitted_curve.param[i])
    coeff = list(coeff)

    residuals = 0
    diff_to_mean = 0
    sum_y = sum(y)
    mean = sum_y / len(y)
    for i in range(0,len(x)):
        if(fitted_curve.type == fitting_types.POLYNOMIAL):
            value = fitted_curve.curve(float(x[i]), coeff)
        else:
            value = fitted_curve.curve(float(x[i]), *coeff)
        value_2 = float(y[i]) - value
        residuals = residuals + math.pow(value_2,2)
        diff_to_mean = diff_to_mean + math.pow(float(y[i]) - mean,2)
    
    r_sqaured_value = 1 - (residuals / diff_to_mean)
    #print(r_sqaured_value)
    fitted_curve.add_r_squared(r_sqaured_value)
    return r_sqaured_value

def all_zero(array):
    zero = True
    for x in array:
        if x != 0:
            return False
    return True

def remove_noise(sheet: dataframe):
    plotting.plot_measurments(sheet)
    fn = sheet.Y
    t = sheet.X
    N = len(t)
    
    h = np.fft.fft(fn ,N)
    PSD = h * np.conj(h)/N
    x_step = 0
    x_max = sheet.X[-1]
    x_min = sheet.X[0]
    x_length = len(sheet.X)
    x_step = (x_max - x_min) / x_length

    freq = []
    for i in range(N):
        freq.append((1 / x_step)*i)

    temp = dataframe("",[freq[0:N//2],PSD[0:N//2], [],[]])
    plotting.plot_line_measurments(temp)

    valid_threshold = False
    threshold = 0
    new_y_data = []
    while(valid_threshold == False):
        threshold = int(input("Noise threshold: "))
        PSD0 = np.where(PSD < threshold, 0, PSD)
        h_prime = np.where(PSD < threshold, 0, h)
        H = np.fft.ifft(h_prime)
        temp_2 = dataframe("",[t,H,[], []])
        plotting.plot_data(sheet, colour = 'k')
        plotting.plot_data(temp_2, colour='r')
        plotting.plot_figure()
        keep = int(input("Keep yes(1) / no(0): "))
        if(keep == 1):
            valid_threshold = True
            for i in range(0,N):
                new_y_data.append(abs(H[i]))
    sheet.Y = new_y_data
    



    

