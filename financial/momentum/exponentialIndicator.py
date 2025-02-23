'''
Exponential Indicator -- Clenow

@author: Manuel Díaz-Meco (manidmt5@gmail.com) 
'''

from financial.strategies.technical.indicator import TechnicalIndicator
from financial.strategies.technical.indicator import Mult

import financial.data as fd


class ExponentialRegressionIndicator(TechnicalIndicator):
    '''
    Exponential Regression-based Momentum Indicator
    '''

    DEFAULT_LOOKAHEAD = 20
    DEFAULT_HORIZON = 90
    MSCI_WORLD = 'URTH'

    def __init__(self, model='exponential'):
        self.lookahead = ExponentialRegressionIndicator.DEFAULT_LOOKAHEAD   # Predicción a 20 días                          | Clenow
        self.horizon = ExponentialRegressionIndicator.DEFAULT_HORIZON       # Ventana de 90 días para entrenar el modelo    | Clenow
        self.model = model

    def get_parameters(self) -> dict:
        '''
        Get indicator parameters
        '''
        parameters = {}
        parameters['lookahead'] = self.lookahead
        parameters['horizon'] = self.horizon
        parameters['model'] = self.model
        return parameters

    def set_parameters(self, parameters: dict) -> TechnicalIndicator:
        '''
        Set indicator parameters
        '''
        try:
            self.lookahead = parameters['lookahead']
        except KeyError:
            self.lookahead = 20

        try:
            self.horizon = parameters['horizon']
        except KeyError:
            self.horizon = 90

        try:
            self.model = parameters['model']
        except KeyError:
            self.model = 'exponential'

        return self


    def get_data_descriptor(self, input_descriptor: fd.DataDescriptor) -> fd.DataDescriptor:
        '''
        Returns a DataDescriptor that accesses the precomputed Beta * R² values.
        '''
        try:
            slope_descriptor = fd.Variable(f"model/momentum/{self.model}/{input_descriptor}@slope")
        except:
            slope_descriptor = fd.Variable(f"model/momentum/{self.model}/{self.MSCI_WORLD}@slope")

        try:
            r2_descriptor = fd.Variable(f"model/momentum/{self.model}/{input_descriptor}@r2")
        except:
            r2_descriptor = fd.Variable(f"model/momentum/{self.model}/{self.MSCI_WORLD}@r2")
        
        return Mult().of([slope_descriptor, r2_descriptor])

