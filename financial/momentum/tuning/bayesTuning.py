'''
Adaptation of Bayesian Optimization for hyperparameter tuning with out-of-sample evaluations.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from financial.lab.tuning.estimators import GaussianProcess, GaussianProcessOptimizer

class OOSBayesGPO(GaussianProcessOptimizer):
    """
    Bayesian Optimization adapted to out-of-sample evaluations.
    """
    def __init__(self, space, gpr: GaussianProcess, restarts: int = 50, weights: dict | None = None):
        super().__init__(space, gpr, restarts)
        # default values:
        self.weights = weights or {"corr": 1.0, "R2": 0.5, "-MAE": 0.25, "-RMSE": 0.25, "hit_rate": 0.0}

    def hyperparameter_score(self, evaluation):
        """
        Compute the score of a hyperparameter configuration based on out-of-sample metrics.
        The score is a weighted sum of selected metrics.
        """
        oos = evaluation["oos"]
        s = 0.0
        for k, w in self.weights.items():
            s += w * float(oos.metric(k))
        return -s


