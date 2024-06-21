
// Demonstration that Stan can't explore modes that are well-separated.

data {
    real loc, scale;
}

parameters {
    real x;
}

model {
    target += log((exp(normal_lpdf(x | 0, scale)) + exp(normal_lpdf(x | loc, scale))) / 2);
}
