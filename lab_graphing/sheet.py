from enum import Enum
import polynomial
class fitting_types(Enum):
    LINEAR          = 0
    POLYNOMIAL      = 1
    COSINE          = 2
    EXPONETIAL      = 3
    LOGARITHMIC_N   = 4
    LOGARITHMIC     = 5
    GUASSIAN        = 6
    NULL            = 7
    CUSTOM          = 8


class fit:
    def __init__(self, fitting_type, x, y, covariance, fitted_curve, num_of_param, param):
        self.type = fitting_type
        self.x = x
        self.y = y
        self.covariance = covariance
        self.curve = fitted_curve
        self.param = param
        self.r_squared = 0
        self.number = num_of_param
    
    def add_r_squared(self, r_squared):
        self.r_squared = r_squared

    def set_to(self, fit_2):
        self.type = fit_2.type
        self.x = fit_2.x
        self.y = fit_2.y
        self.covariance = fit_2.covariance
        self.curve = fit_2.curve
        self.param = fit_2.param
        self.r_squared = fit_2.r_squared
        self.number = fit_2.number

    def print_function(self):
        if(self.type == fitting_types.LINEAR):
            print("{} + {}x".format(self.param[0], self.param[1]))
        elif(self.type == fitting_types.POLYNOMIAL):
            temp = polynomial.Polynomial(len(self.param)-1, 'x', self.param)
            print(temp.print_polynomial())
        elif(self.type == fitting_types.COSINE):
            print("{} + {}sin({}x + {}) + {}cos({}x + {})".format(*self.param))
        elif(self.type == fitting_types.EXPONETIAL):
            print("{} + {}exp({}x + {})".format(*self.param))
        elif(self.type == fitting_types.LOGARITHMIC):
            print("{} + {}log({}x + {}) base {}".format(*self.param))
        elif(self.type == fitting_types.LOGARITHMIC_N):
            print("{} + {}ln({}x + {})".format(*self.param))
        elif(self.type == fitting_types.GUASSIAN):
            print("{} + {}* exp(-0.5 * ((x-{})/{})^2)".format(self.param[0], self.param[3], self.param[2], self.param[1]))
        elif(self.type == fitting_types.CUSTOM):
            print(self.param[0])
            return self.param[0]
        else:
            print("Invalid type")


class dataframe:
    def __init__(self, name, data):
        self.name = name
        self.data_fit = []
        if(len(data) == 4):
            self.X      = data[0]
            self.Y      = data[1]
            self.X_U    = data[2]
            self.Y_U    = data[3]
        else:
            self.X      = []
            self.Y      = []
            self.X_U    = []
            self.Y_U    = []

    def set_to(self, dataframe_2):
        self.name = dataframe_2.name
        self.X      = dataframe_2.X
        self.Y      = dataframe_2.Y
        self.X_U    = dataframe_2.X_U
        self.Y_U    = dataframe_2.Y_U
        self.data_fit = dataframe_2.data_fit
    
    def add_data_fit(self, fit):
        self.data_fit.append(fit)
    



    

