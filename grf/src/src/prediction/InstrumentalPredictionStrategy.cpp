/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "commons/Observations.h"
#include "commons/utility.h"
#include "prediction/InstrumentalPredictionStrategy.h"

const std::size_t InstrumentalPredictionStrategy::OUTCOME = 0;
const std::size_t InstrumentalPredictionStrategy::TREATMENT = 1;
const std::size_t InstrumentalPredictionStrategy::INSTRUMENT = 2;
const std::size_t InstrumentalPredictionStrategy::OUTCOME_INSTRUMENT = 3;
const std::size_t InstrumentalPredictionStrategy::TREATMENT_INSTRUMENT = 4;
const std::size_t InstrumentalPredictionStrategy::Q = 5;
const std::size_t InstrumentalPredictionStrategy::WORIG = 6;
const std::size_t InstrumentalPredictionStrategy::YORIG = 7;
const std::size_t InstrumentalPredictionStrategy::Q2 = 8;
const std::size_t InstrumentalPredictionStrategy::Q3 = 9;
const std::size_t InstrumentalPredictionStrategy::Q4 = 10;

const std::size_t NUM_TYPES = 11;

size_t InstrumentalPredictionStrategy::prediction_length() {
    return 1;
}

std::vector<double> InstrumentalPredictionStrategy::predict(const std::vector<double>& average) {
  double instrument_effect_numerator = average.at(OUTCOME_INSTRUMENT) - average.at(OUTCOME) * average.at(INSTRUMENT);
  double first_stage_numerator = average.at(TREATMENT_INSTRUMENT) - average.at(TREATMENT) * average.at(INSTRUMENT);

  return { instrument_effect_numerator / first_stage_numerator };
}

std::vector<double> InstrumentalPredictionStrategy::compute_variance(
    const std::vector<double>& average,
    const PredictionValues& leaf_values,
    uint ci_group_size) {

  double instrument_effect_numerator = average.at(OUTCOME_INSTRUMENT)
     - average.at(OUTCOME) * average.at(INSTRUMENT);
  double first_stage_numerator = average.at(TREATMENT_INSTRUMENT)
     - average.at(TREATMENT) * average.at(INSTRUMENT);
  double treatment_effect_estimate = instrument_effect_numerator / first_stage_numerator;
  double main_effect = average.at(OUTCOME) - average.at(TREATMENT) * treatment_effect_estimate;

  double num_good_groups = 0;
  std::vector<std::vector<double>> psi_squared = {{0, 0}, {0, 0}};
  std::vector<std::vector<double>> psi_grouped_squared = {{0, 0}, {0, 0}};

  for (size_t group = 0; group < leaf_values.get_num_nodes() / ci_group_size; ++group) {
    bool good_group = true;
    for (size_t j = 0; j < ci_group_size; ++j) {
      if (leaf_values.empty(group * ci_group_size + j)) {
        good_group = false;
      }
    }
    if (!good_group) continue;

    num_good_groups++;

    double group_psi_1 = 0;
    double group_psi_2 = 0;

    for (size_t j = 0; j < ci_group_size; ++j) {

      size_t i = group * ci_group_size + j;
      const std::vector<double>& leaf_value = leaf_values.get_values(i);

      double psi_1 = leaf_value.at(OUTCOME_INSTRUMENT)
                     - leaf_value.at(TREATMENT_INSTRUMENT) * treatment_effect_estimate
                     - leaf_value.at(INSTRUMENT) * main_effect;
      double psi_2 = leaf_value.at(OUTCOME)
                     - leaf_value.at(TREATMENT) * treatment_effect_estimate
                     - main_effect;

      psi_squared[0][0] += psi_1 * psi_1;
      psi_squared[0][1] += psi_1 * psi_2;
      psi_squared[1][0] += psi_2 * psi_1;
      psi_squared[1][1] += psi_2 * psi_2;

      group_psi_1 += psi_1;
      group_psi_2 += psi_2;
    }

    group_psi_1 /= ci_group_size;
    group_psi_2 /= ci_group_size;

    psi_grouped_squared[0][0] += group_psi_1 * group_psi_1;
    psi_grouped_squared[0][1] += group_psi_1 * group_psi_2;
    psi_grouped_squared[1][0] += group_psi_2 * group_psi_1;
    psi_grouped_squared[1][1] += group_psi_2 * group_psi_2;
  }

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      psi_squared[i][j] /= (num_good_groups * ci_group_size);
      psi_grouped_squared[i][j] /= num_good_groups;
    }
  }

  // Using notation from the GRF paper, we want to apply equation (16),
  // \hat{sigma^2} = \xi' V^{-1} Hn V^{-1}' \xi
  // with Hn = Psi as computed above, \xi = (1 0), and
  // V(x) = (E[ZW|X=x] E[Z|X=x]; E[W|X=x] 1).
  // By simple algebra, we can verify that
  // V^{-1}'\xi = 1/(E[ZW|X=x] - E[W|X=x]E[Z|X=x]) (1; -E[Z|X=x]),
  // leading to the expression below for the variance if we
  // use forest-kernel averages to estimate all conditional moments above.

  double avg_Z = average.at(INSTRUMENT);

  double var_between = 1 / (first_stage_numerator * first_stage_numerator)
    * (psi_grouped_squared[0][0]
	     - psi_grouped_squared[0][1] * avg_Z
	     - psi_grouped_squared[1][0] * avg_Z
	     + psi_grouped_squared[1][1] * avg_Z * avg_Z);

  double var_total = 1 / (first_stage_numerator * first_stage_numerator)
    * (psi_squared[0][0]
	     - psi_squared[0][1] * avg_Z
	     - psi_squared[1][0] * avg_Z
	     + psi_squared[1][1] * avg_Z * avg_Z);

  // This is the amount by which var_between is inflated due to using small groups
  double group_noise = (var_total - var_between) / (ci_group_size - 1);

  // A simple variance correction, would be to use:
  // var_debiased = var_between - group_noise.
  // However, this may be biased in small samples; we do an objective
  // Bayes analysis of variance instead to avoid negative values.
  double var_debiased = bayes_debiaser.debias(var_between, group_noise, num_good_groups);

  double variance_estimate = var_debiased;
  return { variance_estimate };
}

size_t InstrumentalPredictionStrategy::prediction_value_length() {
  return NUM_TYPES;
}

PredictionValues InstrumentalPredictionStrategy::precompute_prediction_values(
    const std::vector<std::vector<size_t>>& leaf_samples,
    const Observations& observations) {
  size_t num_leaves = leaf_samples.size();

  std::vector<std::vector<double>> values(num_leaves);

  for (size_t i = 0; i < leaf_samples.size(); ++i) {
    size_t leaf_size = leaf_samples[i].size();
    if (leaf_size == 0) {
      continue;
    }

    std::vector<double>& value = values[i];
    value.resize(NUM_TYPES);

    double sum_Y = 0;
    double sum_W = 0;
    double sum_Z = 0;
    double sum_YZ = 0;
    double sum_WZ = 0;

    double sum_QT = 0; 
    double sum_QC = 0; 
        double sum_Q2T = 0; 
    double sum_Q2C = 0; 
        double sum_Q3T = 0; 
    double sum_Q3C = 0; 
        double sum_Q4T = 0; 
    double sum_Q4C = 0; 
    double sum_YT = 0; 
    double sum_YC = 0; 
    double sum_WORIG = 0; 

    for (auto& sample : leaf_samples[i]) {
      // an obs that falls in the leaf
      sum_Y += observations.get(Observations::OUTCOME, sample); 
      sum_W += observations.get(Observations::TREATMENT, sample);
      sum_Z += observations.get(Observations::INSTRUMENT, sample);
      sum_YZ += observations.get(Observations::OUTCOME, sample) * observations.get(Observations::INSTRUMENT, sample);
      sum_WZ += observations.get(Observations::TREATMENT, sample) * observations.get(Observations::INSTRUMENT, sample);

      sum_QT += observations.get(Observations::Q, sample) * observations.get(Observations::WORIG, sample); 
      sum_QC += observations.get(Observations::Q, sample) * (1 - observations.get(Observations::WORIG, sample)); 

      sum_Q2T += observations.get(Observations::Q2, sample) * observations.get(Observations::WORIG, sample); 
      sum_Q2C += observations.get(Observations::Q2, sample) * (1 - observations.get(Observations::WORIG, sample)); 

      sum_Q3T += observations.get(Observations::Q3, sample) * observations.get(Observations::WORIG, sample); 
      sum_Q3C += observations.get(Observations::Q3, sample) * (1 - observations.get(Observations::WORIG, sample)); 

      sum_Q4T += observations.get(Observations::Q4, sample) * observations.get(Observations::WORIG, sample); 
      sum_Q4C += observations.get(Observations::Q4, sample) * (1 - observations.get(Observations::WORIG, sample)); 

      sum_YT += observations.get(Observations::YORIG, sample) * observations.get(Observations::WORIG, sample); 
      sum_YC += observations.get(Observations::YORIG, sample) * (1 - observations.get(Observations::WORIG, sample)); 
      sum_WORIG += observations.get(Observations::WORIG, sample); 
    }

    value[OUTCOME] = sum_Y / leaf_size;
    value[TREATMENT] = sum_W / leaf_size;
    value[INSTRUMENT] = sum_Z / leaf_size;
    value[OUTCOME_INSTRUMENT] = sum_YZ / leaf_size;
    value[TREATMENT_INSTRUMENT] = sum_WZ / leaf_size;

    value[Q] = (sum_QT / sum_WORIG) -  (sum_QC / (leaf_size - sum_WORIG));
    value[YORIG] = sum_YC / (leaf_size - sum_WORIG); //mean of Y in the control group
    value[WORIG] = sum_WORIG / leaf_size ;
    value[Q2] = (sum_Q2T / sum_WORIG) -  (sum_Q2C / (leaf_size - sum_WORIG));
    value[Q3] = (sum_Q3T / sum_WORIG) -  (sum_Q3C / (leaf_size - sum_WORIG));
    value[Q4] = (sum_Q4T / sum_WORIG) -  (sum_Q4C / (leaf_size - sum_WORIG));
    
  }
  
  return PredictionValues(values, num_leaves, NUM_TYPES);
}

std::vector<double> InstrumentalPredictionStrategy::q_effect(const PredictionValues& leaf_values) {
 size_t num_trees = 0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    num_trees++;
  }
  double q = 0.0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    const std::vector<double>& leaf_value = leaf_values.get_values(n);
    q += leaf_value.at(Q) / num_trees;
  }
  return {q};
}


std::vector<double> InstrumentalPredictionStrategy::q2_effect(const PredictionValues& leaf_values) {
 size_t num_trees = 0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    num_trees++;
  }
  double q = 0.0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    const std::vector<double>& leaf_value = leaf_values.get_values(n);
    q += leaf_value.at(Q2) / num_trees;
  }
  return {q};
}



std::vector<double> InstrumentalPredictionStrategy::q3_effect(const PredictionValues& leaf_values) {
 size_t num_trees = 0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    num_trees++;
  }
  double q = 0.0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    const std::vector<double>& leaf_value = leaf_values.get_values(n);
    q += leaf_value.at(Q3) / num_trees;
  }
  return {q};
}


std::vector<double> InstrumentalPredictionStrategy::q4_effect(const PredictionValues& leaf_values) {
 size_t num_trees = 0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    num_trees++;
  }
  double q = 0.0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    const std::vector<double>& leaf_value = leaf_values.get_values(n);
    q += leaf_value.at(Q4) / num_trees;
  }
  return {q};
}



std::vector<double> InstrumentalPredictionStrategy::yc_effect(const PredictionValues& leaf_values) {

  size_t num_trees = 0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    num_trees++;
  }

  double yc = 0.0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    const std::vector<double>& leaf_value = leaf_values.get_values(n);
    yc += leaf_value.at(YORIG) / num_trees;
  }

  return {yc};
}

std::vector<double> InstrumentalPredictionStrategy::compute_debiased_error(
    size_t sample,
    const std::vector<double>& average,
    const PredictionValues& leaf_values,
    const Observations& observations) {

  double instrument_effect_numerator = average.at(OUTCOME_INSTRUMENT) - average.at(OUTCOME) * average.at(INSTRUMENT);
  double first_stage_numerator = average.at(TREATMENT_INSTRUMENT) - average.at(TREATMENT) * average.at(INSTRUMENT);
  double treatment_effect_estimate = instrument_effect_numerator / first_stage_numerator;

  double outcome = observations.get(Observations::OUTCOME, sample);
  double treatment = observations.get(Observations::TREATMENT, sample);
  double instrument = observations.get(Observations::INSTRUMENT, sample);

  // To justify the squared residual below as an error criterion in the case of CATE estimation
  // with an unconfounded treatment assignment, see Nie and Wager (2017).
  double residual = outcome - (treatment - average.at(TREATMENT)) * treatment_effect_estimate - average.at(OUTCOME);
  double error_raw = residual * residual;

  // Estimates the Monte Carlo bias of the raw error via the jackknife estimate of variance.
  size_t num_trees = 0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    num_trees++;
  }

  // If the treatment effect estimate is due to less than 5 trees, do not attempt to estimate error,
  // as this quantity is unstable due to non-linearities.
  if (num_trees <= 5) {
    return { NAN };
  }

  // Compute 'leave one tree out' treatment effect estimates, and use them get a jackknife estimate of the excess error.
  double error_bias = 0.0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    const std::vector<double>& leaf_value = leaf_values.get_values(n);
    double outcome_loto = (num_trees *  average.at(OUTCOME) - leaf_value.at(OUTCOME)) / (num_trees - 1);
    double treatment_loto = (num_trees *  average.at(TREATMENT) - leaf_value.at(TREATMENT)) / (num_trees - 1);
    double instrument_loto = (num_trees *  average.at(INSTRUMENT) - leaf_value.at(INSTRUMENT)) / (num_trees - 1);
    double outcome_instrument_loto = (num_trees *  average.at(OUTCOME_INSTRUMENT) - leaf_value.at(OUTCOME_INSTRUMENT)) / (num_trees - 1);
    double treatment_instrument_loto = (num_trees *  average.at(TREATMENT_INSTRUMENT) - leaf_value.at(TREATMENT_INSTRUMENT)) / (num_trees - 1);
    double instrument_effect_numerator_loto = outcome_instrument_loto - outcome_loto * instrument_loto;
    double first_stage_numerator_loto = treatment_instrument_loto - treatment_loto * instrument_loto;
    double treatment_effect_estimate_loto = instrument_effect_numerator_loto / first_stage_numerator_loto;
    double residual_loto = outcome - (treatment - treatment_loto) * treatment_effect_estimate_loto - outcome_loto;
    error_bias += (residual_loto - residual) * (residual_loto - residual);
  }
  
  error_bias *= ((num_trees - 1) / num_trees);
  return { error_raw - error_bias };
}
