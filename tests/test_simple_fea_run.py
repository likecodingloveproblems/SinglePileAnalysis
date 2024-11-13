import numpy as np

from fea_model import Pile, CalibrationParams
from materials import SoilProfile, SoilLayer
from matplotlib import pyplot as plt


def test_simple_onill_fea_run():
    soil_profile = SoilProfile(
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
    )
    pile = Pile(
        pile_length=13.1,
        pile_radius=137e-3,
        area=np.pi * (137e-3) ** 2 - np.pi * (137e-3 - 9.3e-3) ** 2,
        soil_profile=soil_profile,
        elasticity_modulus=200e9,
        calibration_params=CalibrationParams(
            Rfb=0.9,
            Sbu=5e-3,
            alpha21=0.01,
            Rfs=1,
        ),
        load=1.1e6,
    )
    pile.analyze()
