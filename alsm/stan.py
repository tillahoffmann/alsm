import numpy as np
import stan


__all__ = [
    'get_samples',
]


FUNCTIONS = {
    # Evaluate the expected connectivity kernel \lambda_{ij} for members i and j of two clusters.
    'evaluate_mean': """
        real evaluate_mean(vector loc1, vector loc2, real scale1, real scale2, real propensity) {
            real d2 = squared_distance(loc1, loc2);
            real var_ = 1 + scale1 ^ 2 + scale2 ^ 2;
            int ndims = num_elements(loc1);
            return propensity ^ 2 * exp(- d2 / (2 * var_)) / var_ ^ (ndims / 2.0);
        }
    """,
    # Evaluate the expected squared connectivity kernel \lambda_{ij}^2 for members i and j of two
    # clusters.
    'evaluate_square': """
        real evaluate_square(vector loc1, vector loc2, real scale1, real scale2, real propensity) {
            real d2 = squared_distance(loc1, loc2);
            real var_ = 1 + 2 * (scale1 ^ 2 + scale2 ^ 2);
            int ndims = num_elements(loc1);
            return propensity ^ 4 * exp(- d2 / var_) / var_ ^ (ndims / 2.0);
        }
    """,
    # Evaluate the expected cross term \lambda_{ij}\lambda_{il} fors members i, j, and l, where i
    # belongs to the first cluster and j and l belong to the second cluster.
    'evaluate_cross': """
        real evaluate_cross(vector loc1, vector loc2, real scale1, real scale2, real propensity) {
            real d2 = squared_distance(loc1, loc2);
            real var_ = 1 + 2 * scale1 ^ 2 + scale2 ^ 2;
            int ndims = num_elements(loc1);
            return propensity ^ 4 * exp(- d2 / var_) / (var_ * (1 + scale2 ^ 2)) ^ (ndims / 2.0);
        }
    """,
    # Evaluate the expected connection vlumes between two clusters. If n2 == 0, we consider the self
    # connection rate.
    'evaluate_aggregate_mean': """
        real evaluate_aggregate_mean(vector loc1, vector loc2, real scale1, real scale2,
                                     real propensity, real n1, real n2) {
            real mean_ = evaluate_mean(loc1, loc2, scale1, scale2, propensity);
            if (n2 > 0) {
                return n1 * n2 * mean_;
            } else {
                return n1 * (n1 - 1) * mean_;
            }
        }
    """,
    # Evaluate the variance of aggregate connection volumes between two clusters. If n2 == 0, we
    # consider the self connection rate.
    'evaluate_aggregate_var': """
        real evaluate_aggregate_var(vector loc1, vector loc2, real scale1, real scale2,
                                    real propensity, real n1, real n2) {
            real y_ij = evaluate_mean(loc1, loc2, scale1, scale2, propensity);
            real y_ijkl = y_ij ^ 2;
            real y_ijji = evaluate_square(loc1, loc2, scale1, scale2, propensity);
            real y_ijij = y_ij + y_ijji;
            real y_ijil = evaluate_cross(loc1, loc2, scale1, scale2, propensity);
            real y_ijkj = evaluate_cross(loc2, loc1, scale2, scale1, propensity);

            // Between group connections.
            if (n2 > 0) {
                return n1 * n2 * (
                    y_ijij
                    + (n2 - 1) * y_ijil
                    + (n1 - 1) * y_ijkj
                    - (n1 + n2 - 1) * y_ijkl
                );
            }
            // Within group connections.
            else {
                return n1 * (n1 - 1) * (
                    y_ijij
                    + y_ijji
                    + 4 * (n1 - 2) * y_ijil
                    - 2 * (2 * n1 - 3) * y_ijkl
                );
            }
        }
    """,
    # Evaluate the concentration parameter phi of the alternative parametrisation of the negative
    # binomial distribution.
    'evaluate_negbinom_2_phi': """
        real evaluate_negbinom_2_phi(real mean, real var_, real epsilon) {
            real invphi = (var_ - mean) / mean ^ 2;
            // If the inverse of the phi parameter in the negbinom_2 parametrisation is negative,
            // the model is underdispersed with respect to a Poisson distribution. This will never
            // be the case in our model, but we may very well end up with negative values due to
            // numerical issues. We'll just return a large, fixed phi.
            return 1 / fmax(invphi, epsilon);
        }
    """,
    # Evaluate a group scale given the fractional mean parameter.
    'evaluate_group_scale': """
        real evaluate_group_scale(real eta, int num_dims) {
            return sqrt((eta ^ (- 2.0 / num_dims) - 1) / 2);
        }
    """,
    # Evaluate the log Jacobian associated with the group scale transformation.
    'evaluate_group_scale_log_jac': """
        real evaluate_group_scale_log_jac(real eta, int num_dims) {
            return - log(eta ^ (- 2.0 / num_dims) - 1) / 2 - (2.0 + num_dims) / num_dims * log(eta);
        }
    """
}
FUNCTIONS['__all__'] = '\n'.join(FUNCTIONS.values())


GROUP_MODEL = """
    functions {
        %(functions)s
    }

    data {
        int<lower=1> num_groups;
        int<lower=1> num_dims;
        int group_adjacency[num_groups, num_groups];
        vector[num_groups] group_sizes;
        real<lower=0> epsilon;
    }

    // Parameters of the model.
    parameters {
        real<lower=0> population_scale;
        vector[num_dims] group_locs[num_groups];
        real<lower=0, upper=1> propensity;
        // This is the fraction of the potential mean within-group connections we can have.
        vector<lower=0, upper=1>[num_groups] eta;
    }

    // Estimate parameters of the negative binomial distribution.
    transformed parameters {
        vector<lower=0>[num_groups] group_scales;
        real mu[num_groups, num_groups];
        real variance[num_groups, num_groups];
        real phi[num_groups, num_groups];

        // Obtain the group scales based on the "fraction of the maximum possible mean".
        for (i in 1:num_groups) {
            group_scales[i] = evaluate_group_scale(eta[i], num_dims);
        }

        for (i in 1:num_groups) {
            for (j in 1:num_groups) {
                mu[i, j] = evaluate_mean(group_locs[i], group_locs[j], group_scales[i],
                                         group_scales[j], propensity);
                // Evaluate within-group connections.
                if (i == j) {
                    mu[i, j] = mu[i, j] * group_sizes[i] * (group_sizes[i] - 1);
                    variance[i, j] = evaluate_aggregate_var(
                        group_locs[i], group_locs[j], group_scales[i], group_scales[j], propensity,
                        group_sizes[i], 0
                    );
                }
                // Evaluate between-group connections.
                else {
                    mu[i, j] = mu[i, j] * group_sizes[i] * group_sizes[j];
                    variance[i, j] = evaluate_aggregate_var(
                        group_locs[i], group_locs[j], group_scales[i], group_scales[j], propensity,
                        group_sizes[i], group_sizes[j]
                    );
                }

                // Threshold the mean and variance such that we don't end up with numerical issues
                // when the connection rate is small.
                mu[i, j] = fmax(mu[i, j], epsilon);
                variance[i, j] = fmax(variance[i, j], epsilon);

                // Evaluate the overdispersion parameter for the negative binomial distribution.
                phi[i, j] = evaluate_negbinom_2_phi(mu[i, j], variance[i, j], epsilon);
            }
        }
    }

    // The actual model.
    model {
        propensity ~ beta(1, 1);
        group_scales ~ cauchy(0, 1);
        population_scale ~ cauchy(0, 1);

        for (i in 1:num_groups) {
            group_locs[i] ~ normal(0, population_scale);
            // Account for the change of variables.
            target += evaluate_group_scale_log_jac(eta[i], num_dims);
            for (j in 1:num_groups) {
                group_adjacency[i, j] ~ neg_binomial_2(mu[i, j], phi[i, j]);
            }
        }
    }

    // Generate posterior predictive replicates.
    generated quantities {
        int ppd_group_adjacency[num_groups, num_groups];
        for (i in 1:num_groups) {
            for (j in 1:num_groups) {
                ppd_group_adjacency[i, j] = neg_binomial_2_rng(mu[i, j], phi[i, j]);
            }
        }
    }
""" % {'functions': FUNCTIONS['__all__']}


def get_samples(fit: stan.fit.Fit, param: str, flatten_chains: bool = True, squeeze: bool = True) \
        -> np.ndarray:
    """
    Get samples from a stan fit.

    Args:
        fit: Stan fit object to get samples from.
        param: Name of the parameter.
        flatten_chains: Whether to combine samples from all chains.
        squeeze: Whether to remove dimensions of unit size.

    Returns:
        Posterior samples with shape `(*param_dims, num_samples * num_chains)` if `flatten_chains`
        is truthy and shape `(*param_dims, num_samples, num_chains)` otherwise.
    """
    samples = fit[param]
    if not flatten_chains:
        *shape, _ = samples.shape
        shape = shape + [-1, fit.num_chains]
        samples = samples.reshape(shape)
    if squeeze:
        samples = np.squeeze(samples)
    return samples
