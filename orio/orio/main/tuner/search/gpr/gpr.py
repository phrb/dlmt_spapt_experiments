import sys, time
import math
import random
import numpy
import scipy.linalg
import orio.main.tuner.search.search
from orio.main.util.globals import *
import copy
import json
import dataset
import os

import rpy2.rinterface as ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import DataFrame, ListVector, IntVector, FloatVector, StrVector, BoolVector, Formula, NULL, r

class GPR(orio.main.tuner.search.search.Search):
    __STEPS              = "steps"
    __STARTING_SAMPLE    = "starting_sample"
    __EXTRA_POINTS       = "extra_points"

    def __init__(self, params):
        self.base      = importr("base")
        self.utils     = importr("utils")
        self.stats     = importr("stats")
        self.algdesign = importr("AlgDesign")
        self.car       = importr("car")
        self.rsm       = importr("rsm")
        self.dplyr     = importr("dplyr")
        self.quantreg  = importr("quantreg")
        self.dicekrig  = importr("DiceKriging")

        # numpy.random.seed(11221)
        # self.base.set_seed(11221)

        self.complete_design_data = None
        self.complete_search_space = None

        self.total_runs = 20
        orio.main.tuner.search.search.Search.__init__(self, params)

        self.name = "GPR"

        self.parameter_ranges = {}

        for i in range(len(self.params["axis_val_ranges"])):
            self.parameter_ranges[self.params["axis_names"][i]] = [0, len(self.params["axis_val_ranges"][i])]

        info("Parameters: " + str(self.parameter_ranges))

        self.parameter_values = {}

        for i in range(len(self.params["axis_val_ranges"])):
            self.parameter_values[self.params["axis_names"][i]] = self.params["axis_val_ranges"][i]

        info("Parameter Range Values: " + str(self.parameter_values))

        self.starting_sample    = 60
        self.steps              = 3
        self.extra_experiments  = 30
        self.testing_set_size   = 150000

        self.__readAlgoArgs()

        self.experiment_data = None
        self.best_points_complete = None

        if self.time_limit <= 0 and self.total_runs <= 0:
            err((
                '%s search requires search time limit or '
                + 'total number of search runs to be defined') %
                self.__class__.__name__)

        self.run_summary_database = dataset.connect("sqlite:///" + 'run_summary.db')
        self.summary = self.run_summary_database["dlmt_run_summary"]

        info("Starting sample: " + str(self.starting_sample))
        info("GPR steps: " + str(self.steps))
        info("Experiments added per step: " + str(self.extra_experiments))

    def generate_valid_sample(self, sample_size):
        search_space_dataframe = {}

        for n in self.axis_names:
            search_space_dataframe[n] = []

        search_space = {}
        evaluated = 0

        info("Generating valid search space of size {0} (does not spend evaluations)".format(sample_size))

        while len(search_space) < sample_size:
            candidate_point      = self.getRandomCoord()
            candidate_point_key  = str(candidate_point)
            evaluated           += 1

            if candidate_point_key not in search_space:
                perf_params = self.coordToPerfParams(candidate_point)

                is_valid = eval(self.constraint, copy.copy(perf_params),
                                dict(self.input_params))

                if is_valid:
                    search_space[candidate_point_key] = candidate_point

                    for n in perf_params:
                        candidate_value = self.parameter_values[n].index(perf_params[n])
                        search_space_dataframe[n].append(candidate_value)

                    if len(search_space) % int(sample_size / 10) == 0:
                        info("Valid coordinates: " + str(len(search_space)) + "/" + str(sample_size))
                        info("Tested coordinates: " + str(evaluated))

                if evaluated % 1000000 == 0:
                    info("Tested coordinates: " + str(evaluated))

        info("Valid/Tested configurations: " + str(len(search_space)) + "/" +
             str(evaluated))

        for k in search_space_dataframe:
            search_space_dataframe[k] = IntVector(search_space_dataframe[k])

        search_space_dataframe_r = DataFrame(search_space_dataframe)
        search_space_dataframe_r = search_space_dataframe_r.rx(StrVector(self.axis_names))

        info("Generated Search Space:")
        info(str(self.utils.str(search_space_dataframe_r)))

        coded_search_space_dataframe_r = self.encode_data(search_space_dataframe_r)

        return coded_search_space_dataframe_r

    def encode_data(self, data):
        formulas = {}

        for parameter in self.parameter_ranges.keys():
            formulas["{0}e".format(parameter)] = Formula("{0}e ~ ({0} - {1}) / {1}".format(parameter, (self.parameter_ranges[parameter][1] - 1.0) / 2.0))

        info("Encoding formulas: " + str(self.utils.str(ListVector(formulas))))
        info("Data Dimensions: " + str(self.base.dim(data)))

        return self.rsm.coded_data(data, formulas = ListVector(formulas))

    def decode_data(self, data):
        formulas = {}

        for parameter in self.parameter_ranges.keys():
            formulas["{0}".format(parameter)] = Formula("{0} ~ round(({0}e * {1}) + {1})".format(parameter, (self.parameter_ranges[parameter][1] - 1.0) / 2.0))

        info("Encoding formulas: " + str(self.utils.str(ListVector(formulas))))
        info("Data Dimensions: " + str(self.base.dim(data)))

        return self.rsm.coded_data(data, formulas = ListVector(formulas))

    def get_design_best(self, design):
        info("Getting Best from Design")
        decoded_design = self.rsm.decode_data(design)

        info("Decoded Design:")
        info(str(decoded_design))

        r_snippet = """library(dplyr)
        data <- %s
        data[data$cost_mean == min(data$cost_mean), ]""" % (decoded_design.r_repr())

        best_line    = robjects.r(r_snippet)
        design_names = [str(n) for n in self.base.names(best_line) if n not in ["cost_mean", "predicted_mean", "predicted_sd", "predicted_mean_2s"]]
        factors      = self.params["axis_names"]

        info("Factors: " + str(factors))
        info("Design Names: " + str(design_names))
        info("Best Design Line: " + str(best_line))

        if type(best_line.rx(1, True)[0]) is int:
            design_line = [v for v in best_line.rx(1, True)]
        else:
            design_line = [int(round(float(v[0]))) for v in best_line.rx(1, True)]

        candidate = [0] * len(factors)

        for i in range(len(design_names)):
            candidate[factors.index(design_names[i])] = design_line[i]

        info("Design Line: ")
        info(str(design_line))

        info("Candidate Line: ")
        info(str(candidate))

        return candidate

    def measure_design(self, encoded_design, step_number):
        design = self.rsm.decode_data(encoded_design)

        info("Measuring design of size " + str(len(design[0])))

        design_names    = [str(n) for n in self.base.names(design) if n not in ["cost_mean", "predicted_mean", "predicted_sd", "predicted_mean_2s"]]
        initial_factors = self.params["axis_names"]
        measurements    = []

        info("Current Design Names: " + str(design_names))

        info("Complete decoded design:")
        info(str(design))

        for line in range(1, len(design[0]) + 1):
            if type(design.rx(line, True)[0]) is int:
                design_line = [v for v in design.rx(line, True)]
            else:
                design_line = [int(round(float(v[0]))) for v in design.rx(line, True)]

            candidate = [0] * len(initial_factors)

            for i in range(len(design_names)):
                candidate[initial_factors.index(design_names[i])] = design_line[i]

            measurement = self.getPerfCosts([candidate])
            if measurement != {}:
                measurements.append(float(numpy.mean(measurement[str(candidate)][0])))
            else:
                measurements.append(robjects.NA_Real)

        encoded_design = encoded_design.rx(True, IntVector(tuple(range(1, len(initial_factors) + 1))))
        encoded_design = self.dplyr.bind_cols(encoded_design, DataFrame({"cost_mean": FloatVector(measurements)}))

        info("Complete design, with measurements:")
        info(str(self.utils.str(encoded_design)))

        encoded_design = encoded_design.rx(self.stats.complete_cases(encoded_design), True)
        encoded_design = encoded_design.rx(self.base.is_finite(self.base.rowSums(encoded_design)), True)

        info("Clean encoded design, with measurements:")
        info(str(self.utils.str(encoded_design)))

        self.utils.write_csv(encoded_design, "design_step_{0}.csv".format(step_number))

        if self.complete_design_data == None:
            self.complete_design_data = encoded_design
        else:
            self.complete_design_data = self.base.rbind(self.complete_design_data, encoded_design)

        return encoded_design

    def gpr(self):
        iterations       = self.steps
        best_value       = float("inf")
        best_point       = []

        training_data = self.generate_valid_sample(self.starting_sample)
        testing_data = self.base.rbind(self.generate_valid_sample(self.testing_set_size),
                                            training_data)

        if self.complete_search_space == None:
            self.complete_search_space = testing_data
        else:
            self.complete_search_space = self.base.rbind(self.complete_search_space,
                                                              testing_data)

        measured_training_data = self.measure_design(training_data, self.current_iteration_id)

        for i in range(iterations):
            self.current_iteration_id = i + 1

            info("Design data:")
            info(str(self.utils.str(self.complete_design_data)))
            info("Complete Search Space:")
            info(str(self.utils.str(self.complete_search_space)))

            self.utils.write_csv(self.complete_design_data, "complete_design_data.csv")
            self.utils.write_csv(self.complete_search_space, "complete_search_space.csv")

            r_snippet = """library(dplyr)
            library(DiceKriging)
            library(rsm)

            training_data <- read.csv("complete_design_data.csv", header = TRUE)
            training_data$X <- NULL
            gpr_model <- km(design = select(training_data, -cost_mean),
                            response = training_data$cost_mean)

            testing_data <- read.csv("complete_search_space.csv", header = TRUE)
            testing_data$X <- NULL
            gpr_prediction <- predict(gpr_model, testing_data, 'UK')

            testing_data$predicted_mean <- gpr_prediction$mean
            testing_data$predicted_sd <- gpr_prediction$sd

            testing_data$predicted_mean_2s <- testing_data$predicted_mean -
                                              (2 * testing_data$predicted_sd)

            gpr_best_points <- arrange(testing_data,
                                       predicted_mean_2s)[1:%s, ]

            gpr_best_points""" %(self.extra_experiments)

            best_predicted_points = robjects.r(r_snippet)
            best_predicted_points = self.rsm.coded_data(best_predicted_points, formulas = self.rsm.codings(self.complete_design_data))

            info("Best Predictions:")
            info(str(best_predicted_points))

            measured_predictions = self.measure_design(best_predicted_points, self.current_iteration_id)

            info("Design data:")
            info(str(self.utils.str(self.complete_design_data)))
            info("Complete Search Space:")
            info(str(self.utils.str(self.complete_search_space)))

        return self.get_design_best(self.complete_design_data), self.starting_sample + (self.steps * self.extra_experiments)

    def searchBestCoord(self, startCoord = None):
        info('\n----- begin GPR -----')

        best_coord     = None
        best_perf_cost = self.MAXFLOAT
        old_perf_cost  = best_perf_cost

        # record the number of runs
        runs = 0
        sruns = 0
        fruns = 0
        start_time = time.time()

        info("Starting GPR")

        best_point, used_points = self.gpr()

        info("Ending GPR")
        info("Best Point: " + str(best_point))

        predicted_best_value = numpy.mean((self.getPerfCosts([best_point]).values()[0])[0])
        starting_point       = numpy.mean((self.getPerfCosts([[0] * self.total_dims]).values()[0])[0])
        speedup              = starting_point / predicted_best_value

        end_time = time.time()
        search_time = start_time - end_time

        info("Speedup: " + str(speedup))

        info('----- end GPR -----')

        info('----- begin GPR summary -----')
        info(' total completed runs: %s' % runs)
        info(' total successful runs: %s' % sruns)
        info(' total failed runs: %s' % fruns)
        info(' speedup: %s' % speedup)
        info('----- end GPR summary -----')

        # return the best coordinate
        return best_point, predicted_best_value, search_time, speedup

    def __readAlgoArgs(self):
        for vname, rhs in self.search_opts.iteritems():
            print vname, rhs
            if vname == self.__STARTING_SAMPLE:
                self.starting_sample= rhs
            elif vname == self.__STEPS:
                self.steps = rhs
            elif vname == self.__EXTRA_EXPERIMENTS:
                self.extra_experiments = rhs
            else:
                err('orio.main.tuner.search.gpr: unrecognized %s algorithm-specific argument: "%s"'
                    % (self.__class__.__name__, vname))
