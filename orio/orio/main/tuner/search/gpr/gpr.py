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
import gc

import rpy2.rinterface as ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import DataFrame, ListVector, IntVector, FloatVector, StrVector, BoolVector, Formula, NULL, r

class GPR(orio.main.tuner.search.search.Search):
    __STEPS              = "steps"
    __STARTING_SAMPLE    = "starting_sample"
    __EXTRA_EXPERIMENTS  = "extra_experiments"
    __TESTING_SET_SIZE   = "testing_set_size"
    __FAILURE_MULTIPLIER = "failure_multiplier"

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
        self.diced     = importr("DiceDesign")

        #numpy.random.seed(11221)
        #self.base.set_seed(11221)

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

        info("Parameter Real Ranges: " + str(self.axis_val_ranges))
        info("Parameter Range Values: " + str(self.parameter_values))

        self.range_matrix = {}

        for i in range(len(self.axis_names)):
            self.range_matrix[self.axis_names[i]] = IntVector(self.axis_val_ranges[i])

        self.range_matrix = ListVector(self.range_matrix)
        info("DataFrame Ranges: " + str(self.base.summary_default(self.range_matrix)))

        self.starting_sample    = int(round(len(self.params["axis_names"]) + 2))
        self.steps              = 22
        self.extra_experiments  = int(round(len(self.params["axis_names"]) * 1))
        self.testing_set_size   = 300000
        self.failure_multiplier = 100

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
        info("Initial Testing Set Size: " + str(self.testing_set_size))
        info("Constraints: " + str(self.constraint))

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
        info(str(self.base.summary_default(search_space_dataframe_r)))

        coded_search_space_dataframe_r = self.encode_data(search_space_dataframe_r)

        return coded_search_space_dataframe_r

    def generate_valid_lhs(self, sample_size):
        # TODO: Expose step size as parameter!
        step_size = 150 * sample_size

        parameters = {}

        info("pkeys: " + str([k for k in self.parameter_ranges.keys()]))
        info("axisnames: " + str([n for n in self.axis_names]))

        for p in self.parameter_ranges.keys():
            parameters[p] = FloatVector([self.parameter_ranges[p][1] - 1.0])

        parameters = DataFrame(parameters)

        info("Computed parameter ranges:")
        info(str(parameters))

        r_snippet = """library(DiceDesign)
        library(stringr)
        library(tibble)

        ranges <- %s
        output <- lhsDesign(n = %s, dimension = %s)
        design <- output$design

        encoded_matrix <- round(design %%*%% diag(ranges))
        mode(encoded_matrix) <- "integer"

        encoded_design <- data.frame(encoded_matrix)
        names(encoded_design) <- names(ranges)

        range_list <- %s

        convert_parameters <- function(name) {
          encoded_design[, name] <- range_list[[name]][encoded_design[, name] + 1]
        }

        converted_design <- data.frame(sapply(names(encoded_design), convert_parameters))

        constraint <- "%s" %%>%%
            str_replace_all(c("True and " = "TRUE & ",
                              "==" = "== ",
                              "<=" = "<= ",
                              "or" = " |",
                              "and" = "&",
                              "\\\\*" = "\\\\* ",
                              " \\\\)" = "\\\\)",
                              "%%" = " %%%% ")) %%>%%
            rlang::parse_expr()

        print(str(constraint))

        valid_design <- converted_design %%>%%
            rownames_to_column() %%>%%
            filter(!!!constraint)

        encoded_design <- encoded_design[valid_design$rowname, ]

        encoded_design[ , %s]""" % (parameters.r_repr(),
                                    step_size,
                                    len(self.axis_names),
                                    self.range_matrix.r_repr(),
                                    self.constraint,
                                    StrVector(self.axis_names).r_repr())

        candidate_lhs = robjects.r(r_snippet)

        info("Candidate LHS:")
        info(str(self.base.summary_default(candidate_lhs)))

        return self.encode_data(candidate_lhs)

    def generate_valid_sobol(self, sample_size, failure_multiplier):
        step_size = failure_multiplier * sample_size

        parameters = {}

        info("pkeys: " + str([k for k in self.parameter_ranges.keys()]))
        info("axisnames: " + str([n for n in self.axis_names]))

        for p in self.parameter_ranges.keys():
            parameters[p] = FloatVector([self.parameter_ranges[p][1] - 1.0])

        parameters = DataFrame(parameters)

        info("Computed parameter ranges:")
        info(str(parameters))

        r_snippet = """library(randtoolbox)
        library(stringr)
        library(tibble)

        ranges <- %s

        sobol_n <- %s
        sobol_dim <- %s

        temp_sobol <- sobol(n = sobol_n,
                            dim = sobol_dim,
                            scrambling = 3,
                            seed = as.integer((99999 - 10000) * runif(1) + 10000),
                            init = TRUE)

        rm(temp_sobol)
        gc()

        design <- sobol(n = sobol_n,
                        dim = sobol_dim,
                        scrambling = 3,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = FALSE)

        encoded_matrix <- round(design %%*%% diag(ranges))
        mode(encoded_matrix) <- "integer"

        encoded_design <- data.frame(encoded_matrix)
        names(encoded_design) <- names(ranges)

        range_list <- %s

        convert_parameters <- function(name) {
          encoded_design[, name] <- range_list[[name]][encoded_design[, name] + 1]
        }

        converted_design <- data.frame(sapply(names(encoded_design), convert_parameters))

        constraint <- "%s" %%>%%
            str_replace_all(c("True and " = "TRUE & ",
                              "==" = "== ",
                              "<=" = "<= ",
                              "or" = " |",
                              "and" = "&",
                              "\\\\*" = "\\\\* ",
                              " \\\\)" = "\\\\)",
                              "%%" = " %%%% ")) %%>%%
            rlang::parse_expr()

        print(str(constraint))

        valid_design <- converted_design %%>%%
            rownames_to_column() %%>%%
            filter(!!!constraint)

        result_design <- encoded_design[valid_design$rowname, ]

        rm(encoded_design)
        rm(converted_design)
        rm(valid_design)
        gc()

        result_design[ , %s]""" % (parameters.r_repr(),
                                   step_size,
                                   len(self.axis_names),
                                   self.range_matrix.r_repr(),
                                   self.constraint,
                                   StrVector(self.axis_names).r_repr())

        candidate_lhs = robjects.r(r_snippet)
        gc.collect()

        info("Valid Design:")
        info(str(self.base.summary_default(candidate_lhs)))

        return self.encode_data(candidate_lhs)

    def encode_data(self, data):
        formulas = {}

        for parameter in self.parameter_ranges.keys():
            formulas["{0}e".format(parameter)] = Formula("{0}e ~ ({0} - {1}) / {1}".format(parameter, (self.parameter_ranges[parameter][1] - 1.0) / 2.0))

        info("Encoding formulas: " + str(self.base.summary_default(ListVector(formulas))))
        info("Data Dimensions: " + str(self.base.dim(data)))

        return self.rsm.coded_data(data, formulas = ListVector(formulas))

    def decode_data(self, data):
        formulas = {}

        for parameter in self.parameter_ranges.keys():
            formulas["{0}".format(parameter)] = Formula("{0} ~ round(({0}e * {1}) + {1})".format(parameter, (self.parameter_ranges[parameter][1] - 1.0) / 2.0))

        info("Encoding formulas: " + str(self.base.summary_default(ListVector(formulas))))
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

        info("Complete original design:")
        info(str(encoded_design))

        for line in range(1, len(design[0]) + 1):
            if type(design.rx(line, True)[0]) is int:
                design_line = [v for v in design.rx(line, True)]
            else:
                design_line = [int(round(float(v[0]))) for v in design.rx(line, True)]

            candidate = [0] * len(initial_factors)

            for i in range(len(design_names)):
            #    if should_redecode:
            #        candidate[initial_factors.index(design_names[i])] = self.parameter_values[design_names[i]].index(design_line[i])
            #    else:
                candidate[initial_factors.index(design_names[i])] = design_line[i]

            info("Evaluating candidate:")
            info(str(candidate))

            measurement = self.getPerfCosts([candidate])
            if measurement != {}:
                measurements.append(float(numpy.mean(measurement[str(candidate)][0])))
            else:
                measurements.append(robjects.NA_Real)

        encoded_design = encoded_design.rx(True, IntVector(tuple(range(1, len(initial_factors) + 1))))
        encoded_design = self.dplyr.bind_cols(encoded_design, DataFrame({"cost_mean": FloatVector(measurements)}))

        info("Complete design, with measurements:")
        info(str(self.base.summary_default(encoded_design)))

        encoded_design = encoded_design.rx(self.stats.complete_cases(encoded_design), True)
        encoded_design = encoded_design.rx(self.base.is_finite(self.base.rowSums(encoded_design)), True)

        info("Clean encoded design, with measurements:")
        info(str(self.base.summary_default(encoded_design)))

        self.utils.write_csv(encoded_design, "design_step_{0}.csv".format(step_number))

        if self.complete_design_data == None:
            self.complete_design_data = encoded_design
        else:
            info(str(self.complete_design_data))
            info(str(encoded_design))

            self.complete_design_data = self.base.rbind(self.complete_design_data, encoded_design)

        return encoded_design

    def validate_sample(self, sample, selected):
        decoded_sample = self.rsm.decode_data(sample)

        parameters = {}

        info("pkeys: " + str([k for k in self.parameter_ranges.keys()]))
        info("axisnames: " + str([n for n in self.axis_names]))

        for p in self.parameter_ranges.keys():
            parameters[p] = FloatVector([self.parameter_ranges[p][1] - 1.0])

        parameters = DataFrame(parameters)

        info("Computed parameter ranges:")
        info(str(parameters))

        r_snippet = """ library(stringr)
        library(tibble)

        ranges <- %s
        print("Ranges:")
        print(str(ranges))

        constraint <- "%s" %%>%%
            str_replace_all(c("True and " = "TRUE & ",
                              "==" = "== ",
                              "<=" = "<= ",
                              "or" = " |",
                              "and" = "&",
                              "\\\\*" = "\\\\* ",
                              " \\\\)" = "\\\\)",
                              "%%" = " %%%% ")) %%>%%
            rlang::parse_expr()

        print(str(constraint))

        sample <- %s
        print(str(sample))

        encoded_matrix <- round(sample)
        print(str(encoded_matrix))

        encoded_design <- data.frame(encoded_matrix)

        range_list <- %s

        convert_parameters <- function(name) {
          encoded_design[, name] <- range_list[[name]][encoded_design[, name] + 1]
        }

        converted_design <- data.frame(sapply(names(encoded_design), convert_parameters))

        valid_sample <- converted_design %%>%%
            rownames_to_column() %%>%%
            filter(!!!constraint)

        valid_sample$rowname""" % (parameters.r_repr(),
                                   self.constraint,
                                   decoded_sample.r_repr(),
                                   self.range_matrix.r_repr())

        valid_sample = robjects.r(r_snippet)
        validated_sample = sample.rx(valid_sample[1:selected], True)
        info("Validated Sample:")
        info(str(self.base.summary_default(validated_sample)))

        self.complete_search_space = self.base.rbind(self.complete_search_space,
                                                     validated_sample)

        return validated_sample

    def gpr(self):
        iterations       = self.steps
        best_value       = float("inf")
        best_point       = []

        training_data = self.generate_valid_sobol(self.starting_sample,
                                                  self.failure_multiplier)

        # TODO: Expose training failure rate as a separate parameter
        testing_data = self.base.rbind(self.generate_valid_sobol(self.testing_set_size,
                                                                 100),
                                       training_data)

        if self.complete_search_space == None:
            self.complete_search_space = testing_data
        else:
            self.complete_search_space = self.base.rbind(self.complete_search_space,
                                                         testing_data)

        measured_training_data = self.measure_design(training_data, self.current_iteration_id)

        for i in range(iterations):
            self.current_iteration_id = i + 1

            self.complete_search_space = self.dplyr.anti_join(self.complete_search_space,
                                                              self.complete_design_data)

            info("Design data:")
            info(str(self.base.summary_default(self.complete_design_data)))
            info("Complete Search Space:")
            info(str(self.base.summary_default(self.complete_search_space)))

            self.utils.write_csv(self.complete_design_data, "complete_design_data.csv")
            self.utils.write_csv(self.complete_search_space, "complete_search_space.csv")

            r_snippet = """library(dplyr)
            library(randtoolbox)
            library(DiceKriging)
            library(DiceOptim)
            library(foreach)
            library(future.apply)
            library(rsm)

            quiet <- function(x) {
              sink(tempfile())
              on.exit(sink())
              invisible(force(x))
            }

            extra_experiments <- %s

            plan(multiprocess, workers = 16)

            training_data <- read.csv("complete_design_data.csv", header = TRUE)
            training_data <- distinct(select(training_data, -X))

            cores <- 8

            gpr_model <- km(design = select(training_data, -cost_mean),
                            response = training_data$cost_mean,
                            multistart = 2 * cores,
                            control = list(pop.size = 400,
                                           BFGSburnin = 500))

            testing_data <- read.csv("complete_search_space.csv", header = TRUE)
            testing_data$X <- NULL

            print("Applying EI to experiments")

            testing_data$expected_improvement <- future_apply(testing_data, 1, EI, gpr_model)

            gpr_best_points <- testing_data %%>%%
              arrange(desc(expected_improvement))

            gpr_best_points <- select(gpr_best_points[1:extra_experiments, ],
                                      -expected_improvement)

            print("Generating perturbation sample")

            gpr_neighbourhood_factor <- 10000
            perturbation_range <- 0.15

            temp_sobol <- sobol(n = extra_experiments * gpr_neighbourhood_factor,
                                dim = length(names(gpr_best_points)),
                                scrambling = 3,
                                seed = as.integer((99999 - 10000) * runif(1) + 10000),
                                init = TRUE)

            rm(temp_sobol)
            quiet(gc())

            perturbation <- sobol(n = extra_experiments * gpr_neighbourhood_factor,
                                  dim = length(names(gpr_best_points)),
                                  scrambling = 3,
                                  seed = as.integer((99999 - 10000) * runif(1) + 10000),
                                  init = FALSE)

            perturbation <- data.frame(perturbation)
            perturbation <- (2 * perturbation_range * perturbation) - perturbation_range

            names(perturbation) <- names(gpr_best_points)

            gpr_selected_neighbourhood <- data.frame(gpr_best_points) %%>%%
                slice(rep(row_number(), gpr_neighbourhood_factor))

            print(str(gpr_selected_neighbourhood))
            print(str(perturbation))
            gpr_selected_neighbourhood <- gpr_selected_neighbourhood + perturbation

            gpr_selected_neighbourhood[gpr_selected_neighbourhood < -1.0] <- -1.0
            gpr_selected_neighbourhood[gpr_selected_neighbourhood > 1.0] <- 1.0

            gpr_sample <- bind_rows(select(testing_data, -expected_improvement),
                                           gpr_selected_neighbourhood) %%>%%
                distinct()

            print(str(gpr_sample))

            gpr_selected_points <- bind_rows(gpr_best_points,
                                             gpr_selected_neighbourhood)

            gpr_best_points <- gpr_best_points %%>%%
                distinct()

            print("Computing perturbed EI")
            gpr_selected_points$expected_improvement <- future_apply(gpr_selected_points,
                                                                     1,
                                                                     EI,
                                                                     gpr_model)

            gpr_selected_points <- gpr_selected_points %%>%%
                arrange(desc(expected_improvement))

            gpr_selected_points <- select(gpr_selected_points,
                                          -expected_improvement) %%>%%
                                   distinct()

            print(str(gpr_selected_points))

            rm(testing_data)
            rm(training_data)
            gc()

            gpr_selected_points""" %(self.extra_experiments)

            best_predicted_points = robjects.r(r_snippet)
            best_predicted_points = self.rsm.coded_data(best_predicted_points, formulas = self.rsm.codings(self.complete_design_data))

            best_predicted_points = self.validate_sample(best_predicted_points,
                                                         self.extra_experiments)

            gc.collect()

            info("Best Predictions:")
            info(str(best_predicted_points))

            measured_predictions = self.measure_design(best_predicted_points, self.current_iteration_id)

            self.complete_search_space = self.dplyr.anti_join(self.complete_search_space,
                                                              self.complete_design_data)

            info("Design data:")
            info(str(self.base.summary_default(self.complete_design_data)))
            info("Complete Search Space:")
            info(str(self.base.summary_default(self.complete_search_space)))

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
            elif vname == self.__TESTING_SET_SIZE:
                self.testing_set_size = rhs
            elif vname == self.__FAILURE_MULTIPLIER:
                self.failure_multiplier = rhs
            else:
                err('orio.main.tuner.search.gpr: unrecognized %s algorithm-specific argument: "%s"'
                    % (self.__class__.__name__, vname))
