
import CSV
import sheet
import data_fitting
import plotting
from polynomial import Polynomial
from sheet import dataframe
from sheet import fit
from sheet import fitting_types
from scipy import integrate as spi

active_sheet = sheet.dataframe('', [])

def menu(sheets):
    i = 1
    print("Sheets currently loaded:")
    for sheet in sheets:
        entry = "({}) {}".format(i, sheet)
        print(entry)
        i = i + 1
    valid_sheet = False
    number = 0
    while(valid_sheet == False):
        number = input("Load sheet {} to {}: ".format(1,i-1))
        try:
            number = int(number)
        except:
            print("Not a number")
            continue
        
        if(number <= i and number >= 1):
            valid_sheet = True
            continue
        print("Invalid sheet number")
    print("Sheet {} loaded sucessfully".format(number))
    active_sheet.set_to(sheets.get(get_key(sheets, number - 1)))


def get_key(dictionary, index):
     for i, key in enumerate(dictionary.keys()):
        if i == index:
            return key

def fitted_function_manipulation_menu(sheet : dataframe):
    exit = False
    active_function : fit = fit(fitting_types.NULL, [], [], [], None, 0, [])
    function_assigned = False
    options = 6
    while(exit == False):
        valid = False
        option = 0
        while(valid == False):
            if(function_assigned == False):
                print("1) Choose function to work on")
                print("2) Exit")
                option = input("Select option (1-2): ")
            else:
                print("1) Choose new function to work on")
                print("2) Fit Gaussian to peak in function")
                print("3) Fit Gaussian to specified range")
                print("4) Integrate under function")
                print("5) Peak of function (if gaussian)")
                print("{}) Exit".format(options))
                option = input("Select option (1-{}): ".format(options))
            
            try:
                option = int(option)
            except:
                print("Invalid input")
                continue
            if(function_assigned == False and option == 2):
                option = options
            if(option < options + 1 or option >= 1):
                valid = True
        if option == 1:
            for model in sheet.data_fit:
                plotting.plot_fit(model)
                option_2 = 0
                print("Operate on function")
                valid = False
                while(valid == False):
                    option_2 = input("Yes(1) / No(0): ")
                    try:
                        option_2 = int(option_2)
                    except:
                        print("Invalid input")
                        continue
                    if(option_2 == 1 or option_2 == 0):
                        valid = True
                    else:
                        print("Input out of range")
                        continue
                if option_2 == 1:
                    active_function.set_to(model)
                    function_assigned = True
                    break
            if function_assigned == True:
                continue
            else:
                print("No function assigned")
        elif option == 2:
            gauss_fit = data_fitting.guassian_peak_fit(model)
            sheet.add_data_fit(gauss_fit)
            continue
        elif option == 3:
            gauss_fit = data_fitting.gaussian_fit_sampe(sheet)
            sheet.add_data_fit(gauss_fit)
        elif option == 4:
            print(spi.simpson(model.y, model.x))
        elif option == 5:
            if model.type == fitting_types.GUASSIAN:
                print(model.param[3])
        elif option == options:
            exit = True
            print("Returning to previous menu")
            continue
        else:
            print("Invalid input")
            continue


def sheet_operations(active_sheet):
    exit = False
    while(exit == False):
        valid = False
        option = 0
        choices = 5
        while(valid == False):
            print("1) Plot data")
            print("2) Fit a function to data")
            print("3) Manipulate fitted functions")
            print("4) Remove noise from data") #uses fast fourier transform to do this
            print("5) Exit")
            option = input("Select option (1-4): ")
            try:
                option = int(option)
            except:
                print("Invalid input")
                continue
            if(option < choices+1 or option >= 1):
                valid = True
        if(option == 1):
            plotting.plotting_menu(active_sheet)
        elif(option == 2):
            data_fitting.fit_function_menu(active_sheet)
        elif(option == 3):
            fitted_function_manipulation_menu(active_sheet)
        elif(option == 4):
            data_fitting.remove_noise(active_sheet)
        elif(option == choices):
            exit = True
            print("Returning to main menu")
        else:
            print("Invalid input")
            continue


def main():
    sheets = None
    quit = False
    ds_loaded = False
    while(quit == False):
        valid = False
        ds_swapped = False
        option = 0
        while(valid == False):
            print("1) Load dataset")
            print("2) Load sheet")
            print("3) Quit")
            option = input("Select option (1-3): ")
            try:
                option = int(option)
            except:
                print("Invalid input")
                continue
            if(option < 4 or option >= 1):
                valid = True
        if(option == 1):
            sheets = CSV.csv_handling()
            ds_loaded = True
            ds_swapped = True
        elif(option == 2 and ds_loaded == True):
            menu(sheets)
            print(active_sheet.name)
            sheet_operations(active_sheet)
        elif(option == 2):
            print("Dataset needs to be loaded")
            continue
        elif(option == 3):
            print("Exiting program...")
            quit = True
        else:
            print("Invalid input")
            continue
        if(quit == True):
            continue
        elif(ds_swapped == True):
            continue



        
    #data_fitting.find_best_fit(active_sheet)
    #data_fitting.polynomial_fit(active_sheet.X, active_sheet.Y, [], test)
    #test_function = lambda x, a, b, c, d, e, f, g: a*x*6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g 
    #data_fitting.non_linear(test_function, active_sheet.X, active_sheet.Y, 7)
    #


main()