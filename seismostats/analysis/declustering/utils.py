# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (C) 2010-2023 GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>
#
# DISCLAIMER
#
# The software Hazard Modeller's Toolkit (hmtk) provided herein
# is released as a prototype implementation on behalf of
# scientists and engineers working within the GEM Foundation (Global
# Earthquake Model).
#
# It is distributed for the purpose of open collaboration and in the
# hope that it will be useful to the scientific, engineering, disaster
# risk and software design communities.
#
# The software is NOT distributed as part of GEM’s OpenQuake suite
# (https://www.globalquakemodel.org/tools-products) and must be considered as a
# separate entity. The software provided herein is designed and implemented
# by scientific staff. It is not developed to the design standards, nor
# subject to same level of critical review by professional software
# developers, as GEM’s OpenQuake software suite.
#
# Feedback and contribution to the software is welcome, and can be
# directed to the hazard scientific staff of the GEM Model Facility
# (hazard@globalquakemodel.org).
#
# The Hazard Modeller's Toolkit (hmtk) is therefore distributed WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# The GEM Foundation, and the authors of the software, assume no
# liability for use of the software.

"""
Utility functions for seismicity calculations
"""
import numpy as np
import pandas as pd


def decimal_year(catalogue: pd.DataFrame) -> np.ndarray:
    """
    Converts the time column of a pandas DataFrame
    to decimal years.
    """
    time = catalogue["time"].astype("datetime64[s]", errors="ignore")
    return (time.dt.year + time.dt.dayofyear / 365).values


def haversine(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    """
    Allows to calculate geographical distance
    using the haversine formula.

    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """
    if not radians:
        cfact = np.pi / 180.0
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (np.sin(dlat / 2.0) ** 2.0) + (
            np.cos(lat1) * np.cos(lat2[i]) * (np.sin(dlon / 2.0) ** 2.0)
        )
        distance[:, i] = (
            2.0 * earth_rad * np.arctan2(np.sqrt(aval), np.sqrt(1 - aval))
        ).T
        i += 1
    return distance
