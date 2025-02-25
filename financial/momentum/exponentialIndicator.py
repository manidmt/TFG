'''
Exponential Indicator -- Clenow

@author: Manuel Díaz-Meco (manidmt5@gmail.com) 
'''

from financial.strategies.technical.indicator import TechnicalIndicator
from financial.strategies.technical.indicator import Wrapper
import os

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
        Returns a DataDescriptor accessing precomputed Beta * R² values.
        Uses MSCI World (URTH) as fallback if specific ticker is not found.
        '''
        

        # Sería mejor acceder a la carpetea model/momentum/{self.model}/{ticker} en vez de los -

        ticker_str = str(input_descriptor)

        slope_wrapper = Wrapper().set_parameters({
            'ticker': f"model-momentum-{self.model}-{ticker_str}@slope",
            'default': f"model-momentum-{self.model}-{self.MSCI_WORLD}@slope"
        })

        r2_wrapper = Wrapper().set_parameters({
            'ticker': f"model-momentum-{self.model}-{ticker_str}@r2",
            'default': f"model-momentum-{self.model}-{self.MSCI_WORLD}@r2"
        })

        composite = fd.Product()
        composite.append(slope_wrapper.get_data_descriptor(input_descriptor))
        composite.append(r2_wrapper.get_data_descriptor(input_descriptor))

        return composite
        
        '''
        ticker_str = str(input_descriptor)

        base_path = f"{os.environ['CACHE']}/model/momentum/{self.model}"
        slope_path = f"{base_path}/{ticker_str}@slope.pkl"
        r2_path = f"{base_path}/{ticker_str}@r2.pkl"

        # Comprobamos explícitamente si los archivos existen
        slope_ticker = (f"model/momentum/{self.model}/{ticker_str}@slope" 
                        if os.path.exists(slope_path) else 
                        f"model/momentum/{self.model}/{self.MSCI_WORLD}@slope")

        r2_ticker = (f"model/momentum/{self.model}/{ticker_str}@r2" 
                    if os.path.exists(r2_path) else 
                    f"model/momentum/{self.model}/{self.MSCI_WORLD}@r2")

        composite = fd.Product()
        composite.append(fd.Variable(slope_ticker))
        composite.append(fd.Variable(r2_ticker))

        return composite
        '''
