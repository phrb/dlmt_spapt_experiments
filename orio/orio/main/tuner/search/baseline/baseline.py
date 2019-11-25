import sys, time, json
import orio.main.tuner.search.search
import numpy
from orio.main.util.globals import *

class Baseline(orio.main.tuner.search.search.Search):
    def __init__(self, params):
        orio.main.tuner.search.search.Search.__init__(self, params)
        # read all algorithm-specific arguments
        self.__readAlgoArgs()

        self.parameter_values = {}

        for i in range(len(self.params["axis_val_ranges"])):
            self.parameter_values[self.params["axis_names"][i]] = self.params["axis_val_ranges"][i]

        info("Parameter Range Values: " + str(self.parameter_values))

    def searchBestCoord(self, startCoord=None):
        info('\n----- begin baseline evaluation -----')

        # Best configurations from data/tests/gpr_sobol_v0
        candidates = [
            {
                "T1_I": self.parameter_values["T1_I"].index(1),
                "T1_J": self.parameter_values["T1_J"].index(2),
                "U_J": self.parameter_values["U_J"].index(1),
                "U_I": self.parameter_values["U_I"].index(14),
                "T2_I": self.parameter_values["T2_I"].index(2048),
                "T2_J": self.parameter_values["T2_J"].index(32),
                "U1_I": self.parameter_values["U1_I"].index(2),
                "OMP": self.parameter_values["OMP"].index(1),
                "VEC1": self.parameter_values["VEC1"].index(1),
                "VEC2": self.parameter_values["VEC2"].index(1),
                "SCR": self.parameter_values["SCR"].index(1),
                "RT_I": self.parameter_values["RT_I"].index(8),
                "RT_J": self.parameter_values["RT_J"].index(1),
            },
            {
                "T1_I": self.parameter_values["T1_I"].index(32),
                "T1_J": self.parameter_values["T1_J"].index(2),
                "U_J": self.parameter_values["U_J"].index(1),
                "U_I": self.parameter_values["U_I"].index(27),
                "T2_I": self.parameter_values["T2_I"].index(1),
                "T2_J": self.parameter_values["T2_J"].index(2),
                "U1_I": self.parameter_values["U1_I"].index(22),
                "OMP": self.parameter_values["OMP"].index(1),
                "VEC1": self.parameter_values["VEC1"].index(1),
                "VEC2": self.parameter_values["VEC2"].index(1),
                "SCR": self.parameter_values["SCR"].index(1),
                "RT_I": self.parameter_values["RT_I"].index(8),
                "RT_J": self.parameter_values["RT_J"].index(1),
            },
            {
                "T1_I": self.parameter_values["T1_I"].index(2048),
                "T1_J": self.parameter_values["T1_J"].index(4),
                "U_J": self.parameter_values["U_J"].index(1),
                "U_I": self.parameter_values["U_I"].index(18),
                "T2_I": self.parameter_values["T2_I"].index(1),
                "T2_J": self.parameter_values["T2_J"].index(8),
                "U1_I": self.parameter_values["U1_I"].index(4),
                "OMP": self.parameter_values["OMP"].index(1),
                "VEC1": self.parameter_values["VEC1"].index(1),
                "VEC2": self.parameter_values["VEC2"].index(1),
                "SCR": self.parameter_values["SCR"].index(1),
                "RT_I": self.parameter_values["RT_I"].index(8),
                "RT_J": self.parameter_values["RT_J"].index(1),
            },
            {
                "T1_I": self.parameter_values["T1_I"].index(64),
                "T1_J": self.parameter_values["T1_J"].index(256),
                "U_J": self.parameter_values["U_J"].index(20),
                "U_I": self.parameter_values["U_I"].index(1),
                "T2_I": self.parameter_values["T2_I"].index(2048),
                "T2_J": self.parameter_values["T2_J"].index(512),
                "U1_I": self.parameter_values["U1_I"].index(30),
                "OMP": self.parameter_values["OMP"].index(1),
                "VEC1": self.parameter_values["VEC1"].index(0),
                "VEC2": self.parameter_values["VEC2"].index(0),
                "SCR": self.parameter_values["SCR"].index(1),
                "RT_I": self.parameter_values["RT_I"].index(8),
                "RT_J": self.parameter_values["RT_J"].index(2),
            },
            {
                "T1_I": self.parameter_values["T1_I"].index(32),
                "T1_J": self.parameter_values["T1_J"].index(1024),
                "U_J": self.parameter_values["U_J"].index(28),
                "U_I": self.parameter_values["U_I"].index(1),
                "T2_I": self.parameter_values["T2_I"].index(2048),
                "T2_J": self.parameter_values["T2_J"].index(1),
                "U1_I": self.parameter_values["U1_I"].index(27),
                "OMP": self.parameter_values["OMP"].index(1),
                "VEC1": self.parameter_values["VEC1"].index(1),
                "VEC2": self.parameter_values["VEC2"].index(0),
                "SCR": self.parameter_values["SCR"].index(1),
                "RT_I": self.parameter_values["RT_I"].index(8),
                "RT_J": self.parameter_values["RT_J"].index(16),
            },
            {
                "T1_I": self.parameter_values["T1_I"].index(2048),
                "T1_J": self.parameter_values["T1_J"].index(2048),
                "U_J": self.parameter_values["U_J"].index(12),
                "U_I": self.parameter_values["U_I"].index(1),
                "T2_I": self.parameter_values["T2_I"].index(2048),
                "T2_J": self.parameter_values["T2_J"].index(1),
                "U1_I": self.parameter_values["U1_I"].index(14),
                "OMP": self.parameter_values["OMP"].index(1),
                "VEC1": self.parameter_values["VEC1"].index(0),
                "VEC2": self.parameter_values["VEC2"].index(1),
                "SCR": self.parameter_values["SCR"].index(1),
                "RT_I": self.parameter_values["RT_I"].index(4),
                "RT_J": self.parameter_values["RT_J"].index(8),
            },
        ]

        candidate_coords = []

        for i in range(len(candidates)):
            candidate_coord = [0] * self.total_dims

            for j in range(len(self.axis_names)):
                candidate_coord[j] = candidates[i][self.axis_names[j]]

            candidate_coords.append(candidate_coord)

        for candidate_coord in candidate_coords:
            candidate_point = numpy.mean((self.getPerfCosts([candidate_coord]).values()[0])[0])

        starting_point = numpy.mean((self.getPerfCosts([[0] * self.total_dims]).values()[0])[0])
        info('----- end baseline evaluation -----')
        # return [[0] * self.total_dims], starting_point, 1, 1.0
        return candidate_coord, candidate_point, 1, starting_point / candidate_point

    def __readAlgoArgs(self):
        info("No args to read")
