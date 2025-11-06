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
from financial.momentum.indicators.exponentialIndicator import ExponentialRegressionIndicator


import os
from dotenv import load_dotenv
from financial.momentum.utilities import find_dotenv

class SimulationFactory:
    """
    Factory for creating and running simulations.
    """
    def __init__(self, start_year, end_year):
        self.start_year = start_year
        self.end_year = end_year

    def run_simulation(self, universe, model_architecture, extra_info, num_assets, active_refuge):
        """
        Run the simulation with the given parameters.
        """
        load_dotenv(dotenv_path=find_dotenv())

        cache = os.environ["CACHE"] + "/"
        model = os.environ["MODEL"]

        ds = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=cache, update_strategy=NoUpdateStrategy()))
        mp = FileModelProvider(model_path=model)
        mc = fm.ModelCache(ds, mp, cache=FileCache(update_strategy=AppendStrategy(), cache_path=model))
        uc = fm.UnifiedCache(ds, mc)

        # Building the main simulation
        refuge = None
        if active_refuge is not None:
            assets = {active_refuge: 1.0}
            refuge = fp.WeightedPortfolio.from_assets("Refuge", assets)
        indicator = ModelIndicator(model_architecture, extra_info)

        filter = fsf.CompositeAssetFilter( [fsf.TopKAssetFilter(k=num_assets), fsf.MinimumValueAssetFilter(threshold=0.0)] )
        allocation = fsa.EqualWeightAllocation()
        strategy = fsr.AssetRankingStrategy("Simulation", universe, indicator, filter, allocation, refuge)

        market = ds.get_data("^GSPC")
        rebalance = fsrebalance.MonthlyRebalancingSchedule(-1, market)

        simulation = fss.StockStrategySimulation(uc, strategy, rebalance, market)
        simulation.verbose = False
        simulation.simulate(start_year=self.start_year, end_year=self.end_year)

        self.modelSimulation = simulation

        # Clenow strategy
        clenow_indicator = ExponentialRegressionIndicator()
        clenow_strategy = fsr.AssetRankingStrategy("Clenow", universe, clenow_indicator, filter, allocation, refuge)
        simulation_clenow = fss.StockStrategySimulation(uc, clenow_strategy, rebalance, market)
        simulation_clenow.verbose = False
        simulation_clenow.simulate(start_year=self.start_year, end_year=self.end_year)

        self.clenowSimulation = simulation_clenow

    def marketBenchmark(self):
        """
        Run the market benchmark simulation.
        """
        ds = fd.CachedDataStore(path=os.environ["DATA"], cache=FileCache(cache_path=os.environ["CACHE"] + "/", update_strategy=NoUpdateStrategy()))
        assets = {'^GSPC': 1.0}
        benchmark = fp.WeightedPortfolio.from_assets("Benchmark", assets)
        monthly_returns = fss.SimulationUtilities.monthly_returns(ds, benchmark, self.start_year, self.end_year)
        returns = {
            "monthly_returns": monthly_returns,
            "cumulative_returns": 100*fps.CumulativeReturn().get_series(monthly_returns)
        }
        return returns
    

    def cumulative_returns(self):
        return 100*fps.CumulativeReturn().get_series(self.modelSimulation.returns())

    def clenow_cumulative_returns(self):
        return 100*fps.CumulativeReturn().get_series(self.clenowSimulation.returns())

    def statistics(self, returns):
        """
        Compute various financial statistics for the given returns.
        """
        start_date = f"{self.start_year}-01-01"
        end_date = f"{self.end_year}-06-30" if self.end_year == 2025 else f"{self.end_year}-12-31"

        statistics = {
             # "average_monthly_rotation": 100*simulation.monthly_rotation(),
             "cumulative_return" : 100*fps.CumulativeReturn().get(returns, start_date, end_date),
             "annualized_return" : 100*fps.AnnualizedReturn().get(returns, start_date, end_date),
             "maximum_drawdown" : 100*fps.MaximumDrawdown().get(returns, start_date, end_date),
             "annualized_volatility" : 100*fps.AnnualizedVolatility().get(returns, start_date, end_date),
             "sharpe_ratio" : fps.SharpeRatio().get(returns, start_date, end_date),
             "sortino_ratio" : fps.SortinoRatio().get(returns, start_date, end_date)
        }

        return statistics

    def simulation_statistics(self):
        """
        Compute various statistics for the simulation.
        """
        statistics = self.statistics(self.modelSimulation.returns())
        statistics["average_monthly_rotation"] = 100*self.modelSimulation.monthly_rotation()
        return statistics

    def clenow_statistics(self):
        """
        Compute various statistics for the Clenow simulation.
        """
        statistics = self.statistics(self.clenowSimulation.returns())
        statistics["average_monthly_rotation"] = 100*self.clenowSimulation.monthly_rotation()
        return statistics

    def benchmark_statistics(self):
        """
        Compute various statistics for the benchmark simulation.
        """
        monthly_returns = self.marketBenchmark()["monthly_returns"]
        statistics = self.statistics(monthly_returns)
        statistics["average_monthly_rotation"] = 0.0
        return statistics

    def all_statistics(self):
        return {
            "simulation": self.simulation_statistics(),
            "clenow": self.clenow_statistics(),
            "benchmark": self.benchmark_statistics()
        }