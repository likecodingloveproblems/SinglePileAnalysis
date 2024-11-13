from typing import Any

import numpy as np
import scipy.optimize
from pydantic import BaseModel
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

from fea_model import Pile, CalibrationParams

x0 = np.array([5 * units["mm"], 0.1, 1.0, 1.0])
ls = optimize.least_squares(
    lsfunc,
    x0,
    bounds=([1.0 * units["mm"], 0.0, 0.8, 0.8], [9 * units["mm"], 1.0, 1.0, 1.0]),
    method="trf",
)

# bounds = [(1*units['mm'],9*units['mm']), (0.0, 1.0), (0.8, 1.0), (0.8, 1.0)]
# de = optimize.differential_evolution(cost, bounds,maxiter = 500, popsize = 20)
# Sbu,k21b,Rfb,Rfs = de.x
# y = func(xdata,Sbu,k21b,Rfb,Rfs)

onill1982_single_pile_force_deformation = np.array(
    [
        [0, 0.45e-3, 1e-3, 1.4e-3, 2e-3, 2.85e-3, 3.4e-3, 4.2e-3],
        [0, 118e3, 250e3, 331e3, 436e3, 550e3, 600e3, 653e3],
    ]
)


class Calibrator(BaseModel):
    pile: Pile
    load_test_results: onill1982_single_pile_force_deformation

    def model_post_init(self, __context: Any) -> None:
        self.pile.load = max(self.load_test_results[1])

    def calibrate(self, method: str):
        """Calibrate parameters to minimize the error of model with respect to pile load test"""
        de = differential_evolution(
            self.cost,
            bounds=self.pile.calibration_params.bounds,
            maxiter=500,
            popsize=20,
        )
        final_calibration_params = CalibrationParams.from_array(de.x)
        return final_calibration_params

    def cost(self, x):
        self.pile.calibration_params = CalibrationParams.from_array(x)
        pile_head_disp, pile_head_force = self.pile.analyze()
        pile_head_force_interp = interp1d(pile_head_disp, pile_head_force, kind='linear', fill_value="extrapolate")
        pile_head_force = pile_head_force_interp(self.load_test_results[0])
        return self.least_square(pile_head_force, self.load_test_results[1])

    def least_square(self, y1, y2):
        return np.sqrt(sum((y2 - y1) ** 2))

