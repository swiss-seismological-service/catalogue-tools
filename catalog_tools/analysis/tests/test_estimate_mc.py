import numpy as np
from numpy.testing import assert_allclose

from catalog_tools.analysis.estimate_mc import empirical_cdf


def test_empirical_cdf():
    sample = np.array(
        [0.8839698, 0.98729853, 0.01606361, 0.20449985, 0.64555887,
         0.51067202, 0.21032875, 0.30893407, 0.01760724, 0.64488555,
         0.52866666, 0.6113724, 0.29912548, 0.88983565, 0.71962906,
         0.95291164, 0.39710704, 0.9886478, 0.45248849, 0.72807291,
         0.34412639, 0.91615602, 0.06441641, 0.00929412, 0.47726263,
         0.47025982, 0.03163823, 0.91309795, 0.93026383, 0.51430296,
         0.54332513, 0.55325218, 0.84463611, 0.91730252, 0.2446928,
         0.90675099, 0.68900707, 0.72307647, 0.29659387, 0.5286394,
         0.17295367, 0.38877039, 0.00549466, 0.15884389, 0.76630311,
         0.51640171, 0.67863794, 0.0685186, 0.86879564, 0.38891402])

    xs = np.array([0.00286059, 0.02645488, 0.09602971, 0.10543411, 0.12122197,
                   0.17101131, 0.18768914, 0.22654742, 0.23244909, 0.25050505,
                   0.2655751, 0.28293811, 0.30126372, 0.30369331, 0.3293481,
                   0.35742368, 0.41904787, 0.41942562, 0.42830868, 0.4500845,
                   0.45608099, 0.5174118, 0.51832513, 0.57837569, 0.59442014,
                   0.60865629, 0.64039346, 0.64907467, 0.6659989, 0.67964183,
                   0.68015698, 0.75870041, 0.76414069, 0.78557629, 0.7916637,
                   0.80366039, 0.82605015, 0.84610349, 0.85449252, 0.86735699,
                   0.87350042, 0.90969489, 0.92326671, 0.94349104, 0.94763226,
                   0.96604459, 0.97282919, 0.97815852, 0.98180271, 0.98602751])
    ys = np.array(
        [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24,
         0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48,
         0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72,
         0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96,
         0.98, 1.])

    x, y = empirical_cdf(sample)

    assert_allclose(x, xs, rtol=1e-7)
    assert_allclose(y, ys, rtol=1e-7)
