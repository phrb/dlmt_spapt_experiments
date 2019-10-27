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

class DLMT(orio.main.tuner.search.search.Search):
    __INTERACTIONS       = "interactions"
    __QUADRATIC          = "quadratic"
    __LINEAR             = "linear"
    __INVERSE            = "inverse"
    __CUBIC              = "cubic"
    __FEDEROV_SAMPLING   = "federov_sampling"
    __STEPS              = "steps"
    __EXTRA_EXPERIMENTS  = "extra_experiments"
    __DESIGN_MULTIPLIER  = "design_multiplier"
    __AOV_THRESHOLD      = "aov_threshold"
    __PREDICTION_USE_ALL = "prediction_use_all"

    def __init__(self, params):
        self.base      = importr("base")
        self.utils     = importr("utils")
        self.stats     = importr("stats")
        self.algdesign = importr("AlgDesign")
        self.car       = importr("car")
        self.rsm       = importr("rsm")
        self.dplyr     = importr("dplyr")
        self.quantreg  = importr("quantreg")

        # numpy.random.seed(11221)
        # self.base.set_seed(11221)

        self.complete_design_data = None
        self.complete_search_space = None

        self.total_runs = 20
        orio.main.tuner.search.search.Search.__init__(self, params)

        self.name = "DLMT"

        self.parameter_ranges = {}

        for i in range(len(self.params["axis_val_ranges"])):
            self.parameter_ranges[self.params["axis_names"][i]] = [0, len(self.params["axis_val_ranges"][i])]

        info("Parameters: " + str(self.parameter_ranges))

        self.parameter_values = {}

        for i in range(len(self.params["axis_val_ranges"])):
            self.parameter_values[self.params["axis_names"][i]] = self.params["axis_val_ranges"][i]

        info("Parameter Range Values: " + str(self.parameter_values))

        self.model              = {}
        self.interactions       = []
        self.quadratic          = []
        self.linear             = []
        self.inverse            = []
        self.cubic              = []
        self.federov_sampling   = 300
        self.steps              = 12
        self.extra_experiments  = 10
        self.design_multiplier  = 1
        self.aov_threshold      = 0.01
        self.prediction_use_all = False

        self.__readAlgoArgs()

        if self.time_limit <= 0 and self.total_runs <= 0:
            err((
                '%s search requires search time limit or '
                + 'total number of search runs to be defined') %
                self.__class__.__name__)

        self.run_summary_database = dataset.connect("sqlite:///" + 'run_summary.db')
        self.summary = self.run_summary_database["dlmt_run_summary"]

        info("Federov Sampling Multiplier: " + str(self.federov_sampling))
        info("ANOVA Steps: " + str(self.steps))
        info("Extra Experiments in Designs: " + str(self.extra_experiments))
        info("Design Multiplier: " + str(self.design_multiplier))
        info("AOV Threshold: " + str(self.aov_threshold))
        info("Use all variables in prediction: " + str(self.prediction_use_all))

    def clean_search_space(self, federov_search_space, full_model):
        data = {}
        r_snippet = """library(AlgDesign)

        data <- %s
        formula <- %s

        model_matrix <- as.data.frame(model.matrix(formula, data))

        one_level_factors <- names(Filter(function(x)(length(unique(x)) < 2), data))
        two_level_factors <- names(Filter(function(x)(length(unique(x)) < 3), data))
        three_level_factors <- names(Filter(function(x)(length(unique(x)) < 4), data))

        one_level_terms <- names(Filter(function(x)(length(unique(x)) < 2), model_matrix))
        two_level_terms <- names(Filter(function(x)(length(unique(x)) < 3), model_matrix))
        three_level_terms <- names(Filter(function(x)(length(unique(x)) < 4), model_matrix))

        list("one_level_factors" = one_level_factors,
             "two_level_factors" = two_level_factors,
             "three_level_factors" = three_level_factors,
             "one_level_terms" = one_level_terms,
             "two_level_terms" = two_level_terms,
             "three_level_terms" = three_level_terms)
        """ % (federov_search_space.r_repr(), Formula(full_model).r_repr())

        output = robjects.r(r_snippet)

        one_level_factors   = [f for f in output[0]]
        two_level_factors   = [f for f in output[1]]
        three_level_factors = [f for f in output[2]]

        one_level_terms   = [f for f in output[3]]
        two_level_terms   = [f for f in output[4]]
        three_level_terms = [f for f in output[5]]

        info("Clean Info:")

        info("One-Level Factors: " + str(one_level_factors))
        info("Two-Level Factors: " + str(two_level_factors))
        info("Three-Level Factors: " + str(three_level_factors))

        info("One-Level Terms: " + str(one_level_terms))
        info("Two-Level Terms: " + str(two_level_terms))
        info("Three-Level Terms: " + str(three_level_terms))

        if one_level_factors:
            self.model["linear"] = [f for f in self.model["linear"] if f not in one_level_factors]
        if one_level_terms:
            self.model["interactions"] = [f for f in self.model["interactions"] if f not in one_level_terms]
        if two_level_factors + two_level_terms:
            clean_two_level_terms = []

            for t in two_level_terms:
                t = t.strip(" ")
                t = t.replace("I(1/(1e-06 + ", "")
                t = t.replace("))", "")
                t = t.replace("I(", "")
                t = t.replace("^2)", "")
                t = t.replace("^3)", "")
                clean_two_level_terms.append(t)

            info("Model Quadratic: " + str(self.model["quadratic"]))
            info("Clean 2 Level Factors/Terms: " + str(two_level_factors + clean_two_level_terms))
            self.model["quadratic"] = [f for f in self.model["quadratic"] if f not in two_level_factors + clean_two_level_terms]
            self.model["inverse"] = [f for f in self.model["inverse"] if f not in two_level_factors + clean_two_level_terms]
        if three_level_factors + three_level_terms:
            clean_three_level_terms = []

            for t in three_level_terms:
                t = t.strip(" ")
                t = t.replace("I(1/(1e-06 + ", "")
                t = t.replace("))", "")
                t = t.replace("I(", "")
                t = t.replace("^2)", "")
                t = t.replace("^3)", "")
                clean_three_level_terms.append(t)


            info("Model Cubic: " + str(self.model["cubic"]))
            info("Clean 3 Level Factors/Terms: " + str(three_level_factors + clean_three_level_terms))
            self.model["cubic"] = [f for f in self.model["cubic"] if f not in three_level_factors + clean_three_level_terms]

        for f in one_level_factors:
            self.model["fixed_factors"][f] = int(federov_search_space.rx(1, f)[0])

        info("Updated Model Info: " + str(self.model))

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

                for k, v in self.model["fixed_factors"].items():
                    perf_params[k] = self.parameter_values[k][int(round(float(v)))]

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

        self.iteration_data["valid_configurations"] = len(search_space)
        self.iteration_data["tested_configurations"] = evaluated

        for k in search_space_dataframe:
            search_space_dataframe[k] = IntVector(search_space_dataframe[k])

        search_space_dataframe_r = DataFrame(search_space_dataframe)
        search_space_dataframe_r = search_space_dataframe_r.rx(StrVector(self.axis_names))

        info("Generated Search Space:")
        info(str(self.utils.str(search_space_dataframe_r)))

        return search_space_dataframe_r

    def opt_monte(self, design_formula, trials, constraint, data):
        info("Starting \"optMonteCarlo\" run")
        info(str(data))

        candidate_multiplier = 10
        repetitions          = 1

        output = self.algdesign.optMonteCarlo(frml        = Formula(design_formula),
                                              data        = data,
                                              constraints = constraint,
                                              nCand       = candidate_multiplier * trials,
                                              nRepeats    = repetitions,
                                              nTrials     = trials)
        return output

    def opt_federov(self, design_formula, trials, data, max_iterations = 1000000, nullify = 0):
        info("Starting \"optFederov\" run")
        info("Using Search Space:")
        info(str(self.utils.str(data)))

        formulas = {}

        for parameter in self.parameter_ranges.keys():
            formulas["{0}e".format(parameter)] = Formula("{0}e ~ ({0} - {1}) / {1}".format(parameter, (self.parameter_ranges[parameter][1] - 1.0) / 2.0))

        info("Encoding formulas: " + str(self.utils.str(ListVector(formulas))))
        info("Data Dimensions: " + str(self.base.dim(data)))

        coded_data = self.rsm.coded_data(data, formulas = ListVector(formulas))

        info("Coded data: " + str(self.utils.str(coded_data)))

        output = self.algdesign.optFederov(frml         = Formula(design_formula),
                                           data         = coded_data,
                                           nTrials      = trials,
                                           nullify      = nullify,
                                           nRepeats     = 10,
                                           maxIteration = max_iterations)

        return output

    def transform_lm(self, design, lm_formula, method = "aov"):
        info("Power Transform Step:")
        response  = lm_formula.split("~")[0].strip()
        variables = lm_formula.split("~")[1].strip()

        info("Current Response: " + str(response))
        info("Current Variables: " + str(variables))
        info("Current Design: " + str(design))

        r_snippet = """design <- %s
        boxcox_t <- powerTransform(%s, data = design)
        regression <- %s(bcPower(%s, boxcox_t$lambda) ~ %s, data = design)
        regression""" %(design.r_repr(), lm_formula, method, response, variables)

        # Catching R runtime errors
        # TODO: Treat errors properly
        try:
            transformed_lm = robjects.r(r_snippet)
            info("Power Transform Successful")
        except:
            transformed_lm = None
            info("Power Transform Unsuccessful")

        return transformed_lm

    def test_heteroscedasticity(self, design, formula, heteroscedasticity_threshold = 0.05):
        lm_regression = self.stats.lm(Formula(formula), data = design)
        heteroscedasticity_test = self.car.ncvTest(lm_regression)

        info("Heteroscedasticity Test p-value: " + str(heteroscedasticity_test.rx("p")[0][0]))

        if heteroscedasticity_test.rx("p")[0][0] < heteroscedasticity_threshold:
            info("Heteroscedasticity transform required.")
            return True
        else:
            info("Heteroscedasticity transform NOT required.")
            return False

    def anova(self, design, formula, heteroscedasticity_threshold = 0.05):
        # Checking for errors in R
        # TODO: Deal better with this, catch actual exceptions
        try:
            info("Anova Formula in Python: " + str(formula))
            info("Anova Formula in R: " + str(Formula(formula)))

            aov_data = self.prune_data(self.complete_design_data)

            if self.test_heteroscedasticity(aov_data, formula, heteroscedasticity_threshold):
                regression = self.transform_lm(aov_data, formula)
            else:
                regression = self.stats.aov(Formula(formula), aov_data)

            if regression == None:
                regression = self.stats.aov(Formula(formula), aov_data)

            summary_regression = self.stats.summary_aov(regression)
            info("Regression Step:" + str(summary_regression))

            prf_values = {}
            for k, v in zip(self.base.rownames(summary_regression[0]), summary_regression[0][4]):
                if k.strip() != "Residuals":
                    prf_values[k.strip()] = v
        except:
            info("Regression Step Failed!")
            regression = None
            prf_values = None

        return regression, prf_values

    def predict_best_values(self, design, formula, size, ordered_prf_keys, prf_values):
        unique_variables = self.get_ordered_fixed_terms(ordered_prf_keys, prf_values)
        info("Predicting Best Values for: " + str(unique_variables))

        if unique_variables == []:
            new_formula = formula
            info("No variables in threshold")
        else:
            new_formula = formula.split("~")[0].strip() + " ~ " + " + ".join(unique_variables)

        self.iteration_data["removed_variables"] = str(unique_variables)

        info("Using Model: " + str(new_formula))
        info("Using Complete Design Data.")

        lm_data = self.prune_data(self.complete_design_data)

        if self.test_heteroscedasticity(lm_data, new_formula):
            regression = self.transform_lm(lm_data, new_formula, method = "lm")
        else:
            regression = self.stats.lm(Formula(new_formula), lm_data)

        if regression == None:
            regression = self.stats.lm(Formula(new_formula), lm_data)

        summary_regression = self.stats.summary_lm(regression)
        info("Prediction Regression Step:" + str(summary_regression))

        new_data = self.generate_valid_sample(size)

        if self.complete_search_space == None:
            self.complete_search_space = new_data
        else:
            self.complete_search_space = self.dplyr.bind_rows(self.complete_search_space, new_data)

        prediction_data = self.prune_data(self.complete_search_space)
        predicted = self.stats.predict(regression, prediction_data)
        predicted_min = min(predicted)

        pruned_data = self.complete_search_space.rx(predicted.ro == self.base.min(predicted), True)

        return pruned_data.rx(1, True)

    def predict_best_values_quantreg(self, design, formula, size, ordered_prf_keys, prf_values, tau = 0.05):
        unique_variables = self.get_ordered_fixed_terms(ordered_prf_keys, prf_values)
        info("Predicting Best Values for: " + str(unique_variables))

        if unique_variables == []:
            new_formula = formula
            info("No variables in threshold")
        else:
            new_formula = formula.split("~")[0].strip() + " ~ " + " + ".join(unique_variables)

        self.iteration_data["removed_variables"] = str(unique_variables)

        info("Using Model: " + str(new_formula))
        info("Using Complete Design Data.")

        lm_data = self.prune_data(self.complete_design_data)

        regression = self.quantreg.rq(Formula(new_formula), tau = tau, data = lm_data)

        summary_regression = self.quantreg.summary_rq(regression)
        info("Prediction Regression Step:" + str(summary_regression))

        new_data = self.generate_valid_sample(size)

        if self.complete_search_space == None:
            self.complete_search_space = new_data
        else:
            self.complete_search_space = self.dplyr.bind_rows(self.complete_search_space, new_data)

        prediction_data = self.prune_data(self.complete_search_space)
        predicted = self.quantreg.predict_rq(regression, prediction_data)
        predicted_min = min(predicted)

        pruned_data = self.complete_search_space.rx(predicted.ro == self.base.min(predicted), True)

        return pruned_data.rx(1, True)

    def predict_best_reuse_data(self, regression, data):
        info("Predicting Best")
        predicted     = self.stats.predict(regression, data)
        predicted_min = min(predicted)
        return data.rx(predicted.ro == self.base.min(predicted), True)

    def predict_best(self, regression, size):
        info("Predicting Best")
        data          = self.generate_valid_sample(size)
        predicted     = self.stats.predict(regression, data)
        predicted_min = min(predicted)
        return data.rx(predicted.ro == self.base.min(predicted), True)

    def get_design_best(self, design):
        info("Getting Best from Design")
        info("Current Model: " + str(self.model))

        info("Design Names: " + str(self.base.names(design)))
        info("Design Response: " + str(design.rx(str(self.model["response"]))))

        if self.base.nrow(design) == 0:
            info("All configurations failed. Returning now.")
            return None

        best_line       = design.rx(design.rx2(str(self.model["response"])).ro == self.base.min(design.rx(str(self.model["response"]))), True)
        design_names    = [str(n) for n in self.base.names(design) if n != self.model["response"]]
        initial_factors = self.params["axis_names"]

        info("Current Design Names: " + str(design_names))
        info("Initial Factors: " + str(initial_factors))

        design_line = [int(round(float(v[0]))) for v in best_line.rx(1, True)]
        candidate   = [0] * len(initial_factors)

        for k, v in self.model["fixed_factors"].items():
            candidate[initial_factors.index(k)] = int(round(float(v)))

        for i in range(len(design_names)):
            candidate[initial_factors.index(design_names[i])] = design_line[i]

        return candidate

    def prune_data(self, data):
        info("Pruning Data for optFederov")

        if self.model["fixed_factors"] == {}:
            pruned_data = data
        else:
            conditions = " & ".join(["{0} == {1}".format(k, v) for k, v in self.model["fixed_factors"].items()])
            r_snippet = """subset(%s, %s)""" % (data.r_repr(), conditions)
            pruned_data = robjects.r(r_snippet)

        info("Pruned data:")
        info(str(self.utils.str(pruned_data)))
        info("Initial data dimensions: " + str(self.base.dim(data)))
        info("Pruned data dimensions: " + str(self.base.dim(pruned_data)))
        return pruned_data

    def get_ordered_fixed_variables(self, ordered_keys, prf_values, threshold = 60):
        info("Getting fixed variables")
        info("Prf Values: ")
        info(str(prf_values))
        info("Ordered Keys: ")
        info(str(ordered_keys))

        unique_variables = []
        for k in ordered_keys:
            # TODO Deal with interactions
            clean_key = k.strip(" ")
            clean_key = clean_key.replace("I(1/(1e-06 + ", "")
            clean_key = clean_key.replace("))", "")
            clean_key = clean_key.replace("I(", "")
            clean_key = clean_key.replace("^2)", "")
            clean_key = clean_key.replace("^3)", "")

            if (clean_key not in unique_variables) and (prf_values[k] < self.aov_threshold):
                unique_variables.append(clean_key)

            if len(unique_variables) >= threshold:
                break

        if unique_variables == []:
            info("No variables within acceptable threshold")
        else:
            info("Variables within threshold: " + str(unique_variables))

        return unique_variables

    def get_ordered_fixed_terms(self, ordered_keys, prf_values, threshold = 60, carry_terms = True):
        info("Getting fixed Model Terms")
        info("Prf Values: ")
        info(str(prf_values))
        info("Ordered Keys: ")
        info(str(ordered_keys))

        unique_variables = []
        for k in ordered_keys:
            if prf_values[k] < self.aov_threshold:
                if carry_terms:
                    clean_key = k.strip(" ")
                    clean_key = clean_key.replace("I(1/(1e-06 + ", "")
                    clean_key = clean_key.replace("))", "")
                    clean_key = clean_key.replace("I(", "")
                    clean_key = clean_key.replace("^2)", "")
                    clean_key = clean_key.replace("^3)", "")

                    for model_term in [term for term in ordered_keys if clean_key in term]:
                        unique_variables.append(model_term)

                else:
                    unique_variables.append(k)

            if len(unique_variables) >= threshold:
                break

        if unique_variables == []:
            info("No variables within acceptable threshold")
        else:
            info("Variables within threshold: " + str(unique_variables))

        return unique_variables

    def get_fixed_variables(self, predicted_best, ordered_prf_keys, prf_values):
        info("Getting fixed variables")
        unique_variables = self.get_ordered_fixed_variables(ordered_prf_keys, prf_values)
        info("Unique Variables: " + str(unique_variables))

        info("Current Model: " + str(self.model))

        for v in unique_variables:
            self.model["fixed_factors"][v] = predicted_best.rx2(1, str(v))[0]

    def prune_model(self, ordered_prf_keys, prf_values):
        info("Pruning Model")
        unique_variables = self.get_ordered_fixed_variables(ordered_prf_keys, prf_values)

        self.model["interactions"] = [f for f in self.model["interactions"] if not f in unique_variables]
        self.model["quadratic"]    = [f for f in self.model["quadratic"] if not f in unique_variables]
        self.model["linear"]       = [f for f in self.model["linear"] if not f in unique_variables]
        self.model["inverse"]      = [f for f in self.model["inverse"] if not f in unique_variables]
        self.model["cubic"]        = [f for f in self.model["cubic"] if not f in unique_variables]

        pruned_interactions = []

        for i in self.model["interactions"]:
            factors = i.split(":")
            selected = True

            for f in factors:
                if f in unique_variables:
                    selected = False
                    break

            if selected:
                pruned_interactions.append(i)

        self.model["interactions"] = pruned_interactions

    def get_federov_data(self, factors):
        low_level_limits  = IntVector([self.parameter_ranges[f][0] for f in factors])
        high_level_limits = IntVector([self.parameter_ranges[f][1] - 1 for f in factors])
        factor_centers    = IntVector([0 for f in factors])
        factor_levels     = IntVector([self.parameter_ranges[f][1] for f in factors])
        factor_round      = IntVector([0 for f in factors])
        is_factor         = BoolVector([False for f in factors])
        mix               = BoolVector([False for f in factors])

        opt_federov_data = {
                             "var": StrVector(factors),
                             "low": low_level_limits,
                             "high": high_level_limits,
                             "center": factor_centers,
                             "nLevels": factor_levels,
                             "round": factor_round,
                             "factor": is_factor,
                             "mix": mix
                           }

        opt_federov_dataframe = DataFrame(opt_federov_data)
        opt_federov_dataframe = opt_federov_dataframe.rx(StrVector(["var",
                                                                   "low",
                                                                   "high",
                                                                   "center",
                                                                   "nLevels",
                                                                   "round",
                                                                   "factor",
                                                                   "mix"]))
        return opt_federov_dataframe

    def get_updated_constraints(self, factors, fixed_variables):
        info("Updating Constraints")
        constraint_text = self.constraint
        variable_ranges = self.parameter_values

        for k, v in fixed_variables.items():
            current_value = str(variable_ranges[k][int(round(float(v)))])
            constraint_text = constraint_text.replace(k, current_value)

        for i in range(len(factors)):
            current_value = ("variable_ranges[\"{0}\"][int(round(float(x[{1}])))]".format(factors[i], i))

            old_value = factors[i]
            constraint_text = constraint_text.replace(old_value, current_value)

        info("Variable Ranges: " + str(variable_ranges))
        info("Current Factors: " + str(factors))
        info("Constraint Text: " + str(self.constraint))

        @ri.rternalize
        def constraint(x):
            type(variable_ranges)
            result = eval(constraint_text)
            return result

        info("Updated Constraint: " + str(constraint_text))
        return constraint

    def measure_design(self, design, encoded_design, step_number):
        info("Measuring design of size " + str(len(design[0])))

        design_names    = [str(n) for n in self.base.names(design)]
        initial_factors = self.params["axis_names"]
        measurements    = []

        info("Current Design Names: " + str(design_names))

        for line in range(1, len(design[0]) + 1):
            if type(design.rx(line, True)[0]) is int:
                design_line = [v for v in design.rx(line, True)]
            else:
                design_line = [int(round(float(v[0]))) for v in design.rx(line, True)]

            candidate = [0] * len(initial_factors)

            for k, v in self.model["fixed_factors"].items():
                candidate[initial_factors.index(k)] = int(round(float(v)))

            for i in range(len(design_names)):
                candidate[initial_factors.index(design_names[i])] = design_line[i]

            measurement = self.getPerfCosts([candidate])
            if measurement != {}:
                measurements.append(float(numpy.mean(measurement[str(candidate)][0])))
            else:
                measurements.append(robjects.NA_Real)

        design = self.base.cbind(design, DataFrame({self.model["response"]: FloatVector(measurements)}))
        encoded_design = self.base.cbind(encoded_design, DataFrame({self.model["response"]: FloatVector(measurements)}))

        info("Complete design, with measurements:")
        info(str(design))

        design = design.rx(self.stats.complete_cases(design), True)
        design = design.rx(self.base.is_finite(self.base.rowSums(design)), True)

        encoded_design = encoded_design.rx(self.stats.complete_cases(encoded_design), True)
        encoded_design = encoded_design.rx(self.base.is_finite(self.base.rowSums(encoded_design)), True)

        info("Clean design, with measurements:")
        info(str(design))

        info("Clean encoded design, with measurements:")
        info(str(encoded_design))

        self.utils.write_csv(encoded_design, "design_step_{0}.csv".format(step_number))
        self.utils.write_csv(design, "decoded_design_step_{0}.csv".format(step_number))

        if self.complete_design_data == None:
            self.complete_design_data = design
        else:
            self.complete_design_data = self.dplyr.bind_rows(self.complete_design_data, design)

        return design

    def dopt_anova_step(self, budget, trials, step_number):
        federov_samples = self.federov_sampling * trials
        prediction_samples = 3 * federov_samples

        new_data = self.generate_valid_sample(federov_samples)

        if self.complete_search_space == None:
            self.complete_search_space = new_data
        else:
            self.complete_search_space = self.dplyr.bind_rows(self.complete_search_space, new_data)

        federov_search_space = self.prune_data(self.complete_search_space)

        full_model = "~ "

        if len(self.model["interactions"]) > 0:
            full_model += " + ".join(self.model["interactions"]) + " + "
        if len(self.model["quadratic"]) > 0:
            full_model += " + ".join(["I({0} ^ 2)".format(f) for f in self.model["quadratic"]]) + " + "
        if len(self.model["linear"]) > 0:
            full_model += " + ".join(self.model["linear"]) + " + "
        if len(self.model["inverse"]) > 0:
            full_model += " + ".join(["I(1 / (1e-06 + {0}))".format(f) for f in self.model["inverse"]]) + " + "
        if len(self.model["cubic"]) > 0:
            full_model += " + ".join(["I({0} ^ 3)".format(f) for f in self.model["cubic"]])

        full_model = full_model.strip(" + ")

        info("Full Model: " + str(full_model))

        self.clean_search_space(federov_search_space, full_model)

        full_model = "~ "

        if len(self.model["interactions"]) > 0:
            full_model += " + ".join(self.model["interactions"]) + " + "
        if len(self.model["quadratic"]) > 0:
            full_model += " + ".join(["I({0} ^ 2)".format(f) for f in self.model["quadratic"]]) + " + "
        if len(self.model["linear"]) > 0:
            full_model += " + ".join(self.model["linear"]) + " + "
        if len(self.model["inverse"]) > 0:
            full_model += " + ".join(["I(1 / (1e-06 + {0}))".format(f) for f in self.model["inverse"]]) + " + "
        if len(self.model["cubic"]) > 0:
            full_model += " + ".join(["I({0} ^ 3)".format(f) for f in self.model["cubic"]])

        full_model = full_model.strip(" + ")

        info("Updated Full Model: " + str(full_model))

        design_formula = full_model
        lm_formula     = self.model["response"] + " " + full_model

        self.iteration_data["model"] = str(full_model)

        info("Current Model: " + str(self.model))
        info("Computing D-Optimal Design")

        info("Computing D-Optimal Design with " + str(trials) + " experiments")
        info("Design Formula: " + str(design_formula))

        encoded_full_model = "~ "

        if len(self.model["interactions"]) > 0:
            encoded_full_model += " + ".join(["{0}e".format(f) for f in self.model["interactions"]]) + " + "
        if len(self.model["quadratic"]) > 0:
            encoded_full_model += " + ".join(["I({0}e ^ 2)".format(f) for f in self.model["quadratic"]]) + " + "
        if len(self.model["linear"]) > 0:
            encoded_full_model += " + ".join(["{0}e".format(f) for f in self.model["linear"]]) + " + "
        if len(self.model["inverse"]) > 0:
            encoded_full_model += " + ".join(["I(1 / (1e-06 + {0}e))".format(f) for f in self.model["inverse"]]) + " + "
        if len(self.model["cubic"]) > 0:
            encoded_full_model += " + ".join(["I({0}e ^ 3)".format(f) for f in self.model["cubic"]])

        encoded_full_model = encoded_full_model.strip(" + ")

        info("Encoded model used in optFederov: " + str(encoded_full_model))

        with open("formula_step_{0}.frml".format(step_number), "w") as frml_file:
            frml_file.write(str(encoded_full_model))

        output = self.opt_federov(encoded_full_model, trials, federov_search_space)
        encoded_design = output.rx("design")[0]
        design = self.rsm.decode_data(encoded_design)

        info(str(design))
        info("D-Efficiency Approximation: " + str(output.rx("D")[0]))

        self.iteration_data["D"] = float(str(output.rx("D")[0]).split(" ")[1])

        design                 = self.measure_design(design, encoded_design, step_number)
        used_experiments       = len(design[0])
        regression, prf_values = self.anova(design, lm_formula)

        design_best = self.get_design_best(design)

        if design_best == None:
            info("All experiments failed. Returning now.")
            return {
                        "prf_values": None,
                        "ordered_prf_keys": None,
                        "design_best": None,
                        "predicted_best": None,
                        "used_experiments": used_experiments
                    }
        elif regression == None or prf_values == None:
            info("Regression failed. Returning design best.")
            return {
                        "prf_values": None,
                        "ordered_prf_keys": None,
                        "design_best": design_best,
                        "predicted_best": None,
                        "used_experiments": used_experiments
                    }
        else:
            ordered_prf_keys = sorted(prf_values, key = prf_values.get)

            if self.prediction_use_all:
                predicted_best = self.predict_best(regression, prediction_samples)

                # Used before to reuse existing sample:
                # predicted_best = self.predict_best_reuse_data(regression, federov_search_space)
            else:
                # Used to generate a new search space and fit a new model with only significant variables:
                # Using classical linear regression:
                # predicted_best = self.predict_best_values(design, lm_formula, prediction_samples, ordered_prf_keys, prf_values)

                # Using quantile regression:
                predicted_best = self.predict_best_values_quantreg(design, lm_formula, prediction_samples, ordered_prf_keys, prf_values)

            self.get_fixed_variables(predicted_best, ordered_prf_keys, prf_values)
            self.prune_model(ordered_prf_keys, prf_values)

            info("Best Predicted: " + str(predicted_best))
            info("Best From Design: " + str(design_best))
            info("Current Model: " + str(self.model))

            self.iteration_data["fixed_factors"] = str(self.model["fixed_factors"])

            return {
                        "prf_values": prf_values,
                        "ordered_prf_keys": ordered_prf_keys,
                        "design_best": design_best,
                        "predicted_best": predicted_best,
                        "used_experiments": used_experiments
                   }

    def dopt_anova(self):
        iterations       = self.steps
        initial_budget   = 1000
        budget           = initial_budget
        used_experiments = 0
        best_value       = float("inf")
        best_point       = []

        for i in range(iterations):
            self.current_iteration_id = i + 1

            if used_experiments >= initial_budget:
                info("Stopping: Used budget")
                break

            self.iteration_data = {"step": i + 1,
                                   "used_budget": used_experiments,
                                   "total_budget": initial_budget}

            model_size = len(self.model["interactions"]) + len(self.model["quadratic"]) + len(self.model["linear"]) + len(self.model["inverse"]) + len(self.model["cubic"])

            self.iteration_data["model_size"] = model_size

            if model_size == 0:
                # TODO: Write self.iteration_data to database with empty values?
                info("Stopping: No more variables in model")
                break

            info("Step {0}".format(i))

            trials = int(round(self.design_multiplier * (len(self.model["interactions"]) + len(self.model["quadratic"]) + len(self.model["linear"]) + len(self.model["inverse"]) + len(self.model["cubic"]))) + self.extra_experiments)

            self.iteration_data["trials"] = trials

            step_data = self.dopt_anova_step(budget, trials, i)

            budget           -= step_data["used_experiments"]
            used_experiments += step_data["used_experiments"]

            starting_point = numpy.mean((self.getPerfCosts([[0] * self.total_dims]).values()[0])[0])
            info("Baseline Point:")
            info(str(starting_point))

            design_best = step_data["design_best"]
            info("Design Best Point:")
            info(str(design_best))

            if design_best == None:
                info("Design completely failed. Skipping this step.")
                design_best_value = best_value
            else:
                design_best_value = numpy.mean((self.getPerfCosts([design_best]).values()[0])[0])

            design_best_slowdown = design_best_value / starting_point
            info("Slowdown (Design Best): " + str(design_best_value / starting_point))

            current_best       = design_best
            current_best_value = design_best_value

            if step_data["predicted_best"] == None:
                info("It seems regression failed. Skipping model update.")
            else:
                predicted_best = [int(round(float(v[0]))) for v in step_data["predicted_best"].rx(1, True)]
                info("Predicted Best Point:")
                info(str(predicted_best))
                info("Length of Predicted Best: " + str(len(predicted_best)))
                info("Original Factor Length: " + str(len(self.params["axis_names"])))
                info("Measuring Predicted Best:")
                info(str(predicted_best))
                measured_predicted_best = self.getPerfCosts([predicted_best])

                if measured_predicted_best != {}:
                    predicted_best_value = numpy.mean((measured_predicted_best.values()[0])[0])
                else:
                    predicted_best_value = float("inf")

                info("Current Model: " + str(self.model))

                predicted_best_slowdown = predicted_best_value / starting_point

                info("Slowdown (Design Best): " + str(design_best_value / starting_point))
                self.iteration_data["design_best"] = design_best_value

                info("Slowdown (Predicted Best): " + str(predicted_best_value / starting_point))
                self.iteration_data["predicted_best"] = predicted_best_value

                info("Budget: {0}/{1}".format(used_experiments, initial_budget))

                if design_best_slowdown < predicted_best_slowdown:
                    info("Best point from design was better than predicted point")
                    current_best = design_best
                    current_best_value = design_best_value
                else:
                    current_best = predicted_best
                    current_best_value = predicted_best_value

            if current_best_value < best_value or best_point == []:
                info("Updating Global Best: " + str(current_best_value))
                best_point = current_best
                best_value = current_best_value

            info("Current Best Point: ")
            info(str(best_point))

            self.iteration_data["current_best"] = best_value
            self.iteration_data["current_best_coordinate"] = str(self.coordToPerfParams(best_point))

            self.summary.insert(self.iteration_data)

        info("Final Best Point: ")
        info(str(best_point))

        return best_point, used_experiments

    def searchBestCoord(self, startCoord = None):
        info('\n----- begin DLMT -----')

        self.model = {
                        "response": "cost_mean",
                        "interactions": self.interactions,
                        "quadratic": self.quadratic,
                        "linear": self.linear,
                        "inverse": self.inverse,
                        "cubic": self.cubic,
                        "fixed_factors": {}
                     }

        info("Initial Model: " + str(self.model))

        best_coord     = None
        best_perf_cost = self.MAXFLOAT
        old_perf_cost  = best_perf_cost

        # record the number of runs
        runs = 0
        sruns = 0
        fruns = 0
        start_time = time.time()

        info("Starting DLMT")

        best_point, used_points = self.dopt_anova()

        info("Ending DLMT")
        info("Best Point: " + str(best_point))

        predicted_best_value = numpy.mean((self.getPerfCosts([best_point]).values()[0])[0])
        starting_point       = numpy.mean((self.getPerfCosts([[0] * self.total_dims]).values()[0])[0])
        speedup              = starting_point / predicted_best_value

        end_time = time.time()
        search_time = start_time - end_time

        info("Speedup: " + str(speedup))

        info('----- end DLMT -----')

        info('----- begin DLMT summary -----')
        info(' total completed runs: %s' % runs)
        info(' total successful runs: %s' % sruns)
        info(' total failed runs: %s' % fruns)
        info(' speedup: %s' % speedup)
        info('----- end DLMT summary -----')

        # return the best coordinate
        return best_point, predicted_best_value, search_time, speedup

    def __readAlgoArgs(self):
        for vname, rhs in self.search_opts.iteritems():
            print vname, rhs
            if vname == self.__INTERACTIONS:
                self.interactions = eval(rhs)
            elif vname == self.__QUADRATIC:
                self.quadratic = eval(rhs)
            elif vname == self.__LINEAR:
                self.linear = eval(rhs)
            elif vname == self.__INVERSE:
                self.inverse = eval(rhs)
            elif vname == self.__CUBIC:
                self.cubic = eval(rhs)
            elif vname == self.__FEDEROV_SAMPLING:
                self.federov_sampling = rhs
            elif vname == self.__STEPS:
                self.steps = rhs
            elif vname == self.__EXTRA_EXPERIMENTS:
                self.extra_experiments = rhs
            elif vname == self.__DESIGN_MULTIPLIER:
                self.design_multiplier = rhs
            elif vname == self.__AOV_THRESHOLD:
                self.aov_threshold = rhs
            elif vname == self.__PREDICTION_USE_ALL:
                self.prediction_use_all = rhs
            elif vname == 'total_runs':
                self.total_runs = rhs
            else:
                err('orio.main.tuner.search.randomsearch: unrecognized %s algorithm-specific argument: "%s"'
                    % (self.__class__.__name__, vname))
