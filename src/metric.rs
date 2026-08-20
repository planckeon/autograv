//! Metric fields. A metric is a function from coordinates to a covariant
//! `(0, 2)` tensor field, generic over the scalar so diffable's jets flow
//! through every layer of differentiation.

use num_traits::{One, Zero, real::Real};

use diffable::traits::{
    Cat, Dual, Euclidean, Field, Right, Sinister, Tensor,
    calculus::{Jet, JetRegion, TensorProduct},
    ι,
};

/// The covariant metric tensor type `g_ij` as `V* ⊗ V*`.
///
/// `Sinister` flips `Dual<V>`'s hand back to `Right` so the balanced tensor
/// product is well-formed; coordinates are identical to `Dual<V> ⊗ Dual<V>`
pub type MetricTensor<V> = TensorProduct<Sinister<Dual<V>>, Dual<V>>;

/// Injects a captured `f64` constant through arbitrarily many jet layers.
///
/// ponytail: mirrors diffable's `ConstantRoute` concept in 8 lines.
/// `Field::Fixed` only reaches one layer deep; this recurses. When `d()`
/// supports metrics, diffable's route machinery replaces this trait.
pub trait ScalarConst: Sized {
    fn from_const(x: f64) -> Self;
}

impl ScalarConst for f64 {
    fn from_const(x: f64) -> Self {
        x
    }
}

impl<𝒞: Cat, S, const N: usize> ScalarConst for Jet<𝒞, S, N>
where
    S: ScalarConst + Field + ι,
    S::C: JetRegion<𝒞>,
{
    fn from_const(x: f64) -> Self {
        Jet::new(S::from_const(x), [S::zero(); N])
    }
}

/// A metric field on an `N`-dimensional coordinate space.
///
/// Implement `g` once, generically over `V`; the same body evaluates at real
/// points and at jet-valued points during differentiation.
pub trait MetricField<const N: usize> {
    fn g<V>(&self, x: V) -> MetricTensor<V>
    where
        V: Euclidean + Tensor<Hand = Right>,
        V::F: Real + ScalarConst;
}

/// Minkowski metric, signature `(−, +, +, +)`, constant on `R⁴`.
pub struct Minkowski;

impl MetricField<4> for Minkowski {
    fn g<V>(&self, x: V) -> MetricTensor<V>
    where
        V: Euclidean + Tensor<Hand = Right>,
        V::F: Real + ScalarConst,
    {
        let _ = x;
        TensorProduct::from_fn_ij(|i, j| {
            if i == j {
                if i == 0 { -V::F::one() } else { V::F::one() }
            } else {
                V::F::zero()
            }
        })
    }
}

/// Standard metric of the 2-sphere in `(r, θ, φ)`.
/// i.e., flat three-dimensional Euclidean metric in spherical coordinates
pub struct SphericalPolar;

impl MetricField<3> for SphericalPolar {
    fn g<V>(&self, x: V) -> MetricTensor<V>
    where
        V: Euclidean + Tensor<Hand = Right>,
        V::F: Real + ScalarConst,
    {
        let r = x[0];
        let r2 = r * r;
        let s2 = r2 * x[1].sin() * x[1].sin();
        TensorProduct::from_fn_ij(|i, j| {
            if i == j {
                match i {
                    0 => V::F::one(),
                    1 => r2,
                    _ => s2,
                }
            } else {
                V::F::zero()
            }
        })
    }
}

/// Schwarzschild metric in `(t, r, θ, φ)`
/// - `g00 = −(1 − rs/r)·c²`
/// - `g11 = 1/(1 − rs/r)`
/// - `g22 = r²`
/// - `g33 = r² sin²θ`
pub struct Schwarzschild {
    /// Schwarzschild radius `2GM/c²` in the chosen unit system.
    pub rs: f64,
    /// Speed of light, as used in `g00`.
    pub c: f64,
}

impl MetricField<4> for Schwarzschild {
    fn g<V>(&self, x: V) -> MetricTensor<V>
    where
        V: Euclidean + Tensor<Hand = Right>,
        V::F: Real + ScalarConst,
    {
        let r = x[1];
        let f = V::F::one() - V::F::from_const(self.rs) / r;
        let c2 = V::F::from_const(self.c * self.c);
        let r2 = r * r;
        let s2 = r2 * x[2].sin() * x[2].sin();
        TensorProduct::from_fn_ij(|i, j| {
            if i == j {
                match i {
                    0 => -f * c2,
                    1 => V::F::one() / f,
                    2 => r2,
                    _ => s2,
                }
            } else {
                V::F::zero()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffable::coords::Coords;

    #[test]
    fn minkowski_signature() {
        let m = Minkowski;
        let g = m.g(Coords([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(
            (g[(0, 0)], g[(1, 1)], g[(2, 2)], g[(3, 3)]),
            (-1.0, 1.0, 1.0, 1.0)
        );
        assert_eq!(g[(0, 1)], 0.0);
    }

    #[test]
    fn spherical_polar_diagonal() {
        let m = SphericalPolar;
        let x = Coords([5.0, std::f64::consts::FRAC_PI_3, 0.0]);
        let g = m.g(x);
        let r2 = 25.0;
        let s2 = r2 * (std::f64::consts::FRAC_PI_3.sin()).powi(2);

        assert_eq!(g[(0, 0)], 1.0);
        assert!((g[(1, 1)] - r2).abs() < 1e-12);
        assert!((g[(2, 2)] - s2).abs() < 1e-12);
        assert_eq!(g[(1, 2)], 0.0);
    }
}
