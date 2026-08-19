//! Metric fields. A metric is a function from coordinates to a covariant
//! `(0, 2)` tensor field, generic over the scalar so diffable's jets flow
//! through every layer of differentiation.

use num_traits::{One, Zero, real::Real};

use diffable::traits::{
    Cat, Dual, Euclidean, Field, Right, Sinister, Tensor,
    calculus::{Jet, JetRegion, TensorProduct},
    ι,
};
