import sys, time, json
import orio.main.tuner.search.search
import numpy
from orio.main.util.globals import *

class Baseline(orio.main.tuner.search.search.Search):
    def __init__(self, params):
        orio.main.tuner.search.search.Search.__init__(self, params)
        # read all algorithm-specific arguments
        self.__readAlgoArgs()

    def searchBestCoord(self, startCoord=None):
        info('\n----- begin baseline evaluation -----')
        starting_point = numpy.mean((self.getPerfCosts([[0] * self.total_dims]).values()[0])[0])
        info('----- end baseline evaluation -----')
        return [[0] * self.total_dims], starting_point, 1, 1.0

    def __readAlgoArgs(self):
        info("No args to read")
