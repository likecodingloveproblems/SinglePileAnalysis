import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from fea_model import Pile, CalibrationParams
from materials import SoilProfile, SoilLayer

onill1982_single_pile_force_deformation = np.array(
    [
        [0, 0.45e-3, 1e-3, 1.4e-3, 2e-3, 2.85e-3, 3.4e-3, 4.2e-3],
        [0, 118e3, 250e3, 331e3, 436e3, 550e3, 600e3, 653e3],
    ]
)


def test_onill_1982():
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
        elasticity_modulus=210e9,
        calibration_params=CalibrationParams(
            Rfb=1,
            Sbu=5e-3,
            alpha21=0.01,
            Rfs=1,
        ),
        load=653e3,
    )
    result = pile.analyze()
    plt.figure()
    plt.plot(result[0], result[1], label="fea")
    plt.plot(
        onill1982_single_pile_force_deformation[0],
        onill1982_single_pile_force_deformation[1],
        label="o'nill 1982",
    )
    plt.legend()
    plt.show(block=False)
    time.sleep(1)
    plt.close()
