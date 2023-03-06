import math
from types import FunctionType

class Polynomial:
    def __init__(self,order, variable, coeff):
        self.order = order + 1
        self.variable = variable
        self.coeff = coeff

    def return_value(self, value):
        r_value = 0
        index = 0
        for i in self.coeff:
            index = index + 1
            r_value = r_value + (i*(math.pow(value,self.order-index)))
        return r_value

    def print_polynomial(self):
        items = []
        for i, x in enumerate(reversed(self.coeff)):
            if not x:
                continue
            items.append('{}{}^{}'.format(x if x != 1 else '', self.variable,i))
        items.reverse()
        output = ' + '.join(items)
        output = output.replace('{}^0'.format(self.variable), '')
        output = output.replace('^1 ', ' ')
        output = output.replace('+ -', '- ')

        return output

    def get_order(self):
        return self.order - 1

def evaluate(x, coeff):
    value = 0
    index = 0
    order = len(coeff)
    for i in coeff:
        index = index + 1
        value = value + (i * math.pow(x, order - index))
    return value