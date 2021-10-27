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
}
FUNCTIONS['__all__'] = '\n'.join(FUNCTIONS.values())
