'''
General Model Indicator Class

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from financial.strategies.technical.indicator import TechnicalIndicator
from financial.data import Variable


class ModelIndicator(TechnicalIndicator):

    def __init__(self, model):
        self.model = model

    
    def get_data_descriptor(self, input_descriptor):
        ticker = str(input_descriptor)
        ticker_model = f"{self.model}_{ticker}_2025_single"
        return Variable(ticker_model)