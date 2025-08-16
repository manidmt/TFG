'''
Simulation Factory for Financial Strategies with Models created

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

import financial.data as fd
import financial.model as fm
import financial.portfolio as fp

import financial.strategies.simulation as fss
import financial.strategies.allocation as fsa
import financial.strategies.filter as fsf
import financial.strategies.rank as fsr
import financial.portfolios.statistics as fps
import financial.strategies.rebalance as fsrebalance

from financial.io.file.cache import FileCache
from financial.io.cache import AppendStrategy
from financial.io.file.model import FileModelProvider
from financial.io.cache import NoUpdateStrategy
from financial.momentum.indicators.modelIndicator import ModelIndicator


import os
from dotenv import load_dotenv
from financial.momentum.utilities import find_dotenv

class SimulationFactory:
    def __init__(self, start_year, end_year):
        self.start_year = start_year
        self.end_year = end_year

    def run_simulation(self, universe, model_architecture, extra_info, num_assets, active_refuge):
        
        load_dotenv(dotenv_path=find_dotenv())

        cache = os.environ["CACHE"] + "/"
        model = os.environ["MODEL"]

        ds = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=cache, update_strategy=NoUpdateStrategy()))
        mp = FileModelProvider(model_path=model)
        mc = fm.ModelCache(ds, mp, cache=FileCache(update_strategy=AppendStrategy(), cache_path=model))
        uc = fm.UnifiedCache(ds, mc)
        if active_refuge is None:
                refuge = None
        indicator = ModelIndicator(model_architecture, extra_info)

        filter = fsf.CompositeAssetFilter( [fsf.TopKAssetFilter(k=num_assets), fsf.MinimumValueAssetFilter(threshold=0.0)] )
        allocation = fsa.EqualWeightAllocation()
        strategy = fsr.AssetRankingStrategy("Simulation", universe, indicator, filter, allocation, refuge)

        benchmark = fp.BenchmarkPortfolio("^GSPC")
        market = ds.get_data("^GSPC")
        rebalance = fsrebalance.MonthlyRebalancingSchedule(-1, market)

        simulation = fss.StockStrategySimulation(uc, strategy, rebalance, market)
        simulation.verbose = False
        simulation.simulate(start_year=self.start_year, end_year=self.end_year)

        self.modelSimulation = simulation
        return simulation
    
    def cumulative_returns(self):
        return fps.CumulativeReturn().get_series(self.modelSimulation.returns())

    def portfolio_statistics(self):
        start_date = f"{self.start_year}-01-01"
        end_date = f"{self.end_year}-12-31"
        
        statistics = {
             "average_monthly_rotation": 100*self.modelSimulation.monthly_rotation(),
             "cumulative_return" : 100*fps.CumulativeReturn().get(self.modelSimulation.returns()),
             "annualized_return" : 100*fps.AnnualizedReturn().get(self.modelSimulation.returns(), start_date, end_date),
             "maximum_drawdown" : 100*fps.MaximumDrawdown().get(self.modelSimulation.returns()),
             "annualized_volatility" : 100*fps.AnnualizedVolatility().get(self.modelSimulation.returns()),
             "sharpe_ratio" : fps.SharpeRatio().get(self.modelSimulation.returns()),
             "sortino_ratio" : fps.SortinoRatio().get(self.modelSimulation.returns())
        }

        return statistics