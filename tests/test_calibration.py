from matplotlib import pyplot as plt

from calibrator import Calibrator
from fea_model import CalibrationParams
from load_tests import LoadTestResult


def test_onill_1982():
    load_test_result = LoadTestResult.onill_1982_single_pile()
    calibrator = Calibrator(load_test_result=load_test_result)
    final_calibration_params = calibrator.calibrate()
    pile = load_test_result.get_pile(final_calibration_params)
    calibrated_result = pile.analyze()
    pile = load_test_result.get_pile(CalibrationParams.from_default())
    result = pile.analyze()
    print(f"calibrated params: {final_calibration_params}")
    plt.figure()
    plt.plot(calibrated_result[0], calibrated_result[1], label="calibrated FEA")
    plt.plot(result[0], result[1], label="FEA")
    plt.plot(
        load_test_result.displacements,
        load_test_result.forces,
        label="o'nill 1982",
    )
    plt.legend()
    plt.show()
