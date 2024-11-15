from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from fea_model import CalibrationParams, Pile
from materials import SoilProfile, SoilLayer


class LoadTestResult(BaseModel):
    displacements: np.ndarray
    forces: np.ndarray
    pile_length: float
    pile_radius: float
    pile_area: float | None = None
    pile_elasticity_modulus: float
    soil_profile: SoilProfile
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.pile_area is None:
            self.pile_area = np.pi * self.pile_radius**2

    @property
    def max_load(self):
        return max(self.forces)

    def get_pile(self, calibration_params: CalibrationParams):
        return Pile(
            pile_length=self.pile_length,
            pile_radius=self.pile_radius,
            area=self.pile_area,
            soil_profile=self.soil_profile,
            elasticity_modulus=self.pile_elasticity_modulus,
            calibration_params=calibration_params,
            load=self.max_load,
        )

    @staticmethod
    def onill_1982_single_pile():
        return LoadTestResult(
            displacements=np.array(
                [0, 0.45e-3, 1e-3, 1.4e-3, 2e-3, 2.85e-3, 3.4e-3, 4.2e-3]
            ),
            forces=np.array([0, 118e3, 250e3, 331e3, 436e3, 550e3, 600e3, 653e3]),
            pile_length=13.1,
            pile_radius=137e-3,
            pile_area=np.pi * 137e-3**2 - np.pi * (137e-3 - 9.3e-3) ** 2,
            pile_elasticity_modulus=210e9,
            soil_profile=SoilProfile(
                layers=[
                    SoilLayer(
                        up_depth=0,
                        bottom_depth=13.1,
                        up_shear_modulus=65e6,
                        bottom_shear_modulus=65e6,
                        up_poisson_ratio=0.5,
                        bottom_poisson_ratio=0.5,
                        up_tau_f=19e3,
                        bottom_tau_f=93e3,
                    ),
                ]
            ),
        )
