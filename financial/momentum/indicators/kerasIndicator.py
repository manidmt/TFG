'''
Indicator class for Keras-based financial momentum indicators.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from financial.strategies.technical.indicator import TechnicalIndicator
from financial.strategies.technical.indicator import Wrapper
import financial.data as fd
import pandas as pd
from financial.data import Variable

class KerasIndicator(TechnicalIndicator):
    '''
    Keras-based Momentum Indicator
    '''

    def __init__(self, model):
        self.model = model

    def get_parameters(self) -> dict:
        '''
        Get indicator parameters
        '''
        parameters = {}
        parameters['model'] = self.model
        return parameters

    def get_data(self, provider: fd.DataStore, ticker: str, start_index: str = None, end_index: str = None) -> pd.Series:
        """
        Get predicted data from the model wrapper for the given ticker
        """
        descriptor = self.get_data_descriptor_for_ticker(ticker)
        series = descriptor.get_data(provider, start_index, end_index)
        
        # Asegurarse de que es una serie plana
        if isinstance(series, pd.Series):
            return pd.Series(series, name=f"{ticker}_{self.__class__.__name__}")
        
        # Si por alguna razón el wrapper devolviera un dict o tuple
        if isinstance(series, tuple):
            series = series[0]
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected Series, got {type(series)} from descriptor.get_data")
        
        return pd.Series(series, name=f"{ticker}_{self.__class__.__name__}")

    def get_data_descriptor(self, input_descriptor: fd.DataDescriptor) -> fd.DataDescriptor:
        '''
        Get the data descriptor for the Keras model
        '''
        ticker = str(input_descriptor)
        ticker_model = f"model-momentum-{self.model}_{ticker}_2025_single@pred"
        wrapper = Wrapper().set_parameters({
            'ticker': ticker_model,
            'default': ticker_model
        })

        return wrapper.get_data_descriptor(input_descriptor)
