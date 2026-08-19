//! Forward-mode Jacobians over diffable's truncated Taylor jets.
//!
//! `diffable`'s public `d(f)` API currently only composes programs whose
//! output tensor type is presentation-independent (self-maps `V -> V`, see
//! `JetMap`'s blanket `Fn` impl in `src/traits/calculus.rs`). A metric
//! `x -> V* ⊗ V*` does not fit that blanket impl, so this module performs the
//! same seeding step `d::at` runs internally — seed a unit jet per input
//! coordinate, evaluate, read the first Taylor coefficient — reusing
//! diffable's `Jet` arithmetic verbatim.
//!
//! ponytail: ~25 lines instead of a custom AD engine; when diffable supports
//! general tensor-valued `JetMap`, replace this module's body with `d(f).at(x)`
//! and delete it — the GR layer will not change.

use num_traits::{One, Zero, real::Real};

use diffable::traits::{
    Euclidean, Tensor,
    calculus::{Jet, JetRegion, JetVector},
    ι, 𝐑𝐞𝐚𝐥,
};

/// Jacobian of `f` at `point`, in **output-major** flat order:
/// `out[o * V::N + i] = ∂f_o / ∂x_i`.
///
/// `V` is any Euclidean presentation (plain `Coords<f64, N>` or a nested jet
/// presentation like `JetVector<…, Coords<…>, 1, Jet<…>>`), so the same
/// function differentiates at any depth — first derivatives of the metric,
/// and (via the closure calling `christoffel_generic`) derivatives of the
/// connection.
pub fn jacobian_of<V, F, G>(f: F, point: &V) -> Vec<V::F>
where
    V: Euclidean,
    V::F: Real + ι<C: JetRegion<𝐑𝐞𝐚𝐥::𝒞>>,
    F: Fn(JetVector<𝐑𝐞𝐚𝐥::𝒞, V, 1, V::F>) -> G,
    G: Tensor<F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F, 1>>,
{
    let mut out = vec![V::F::zero(); G::N * V::N];
    for i in 0..V::N {
        let input = JetVector::<𝐑𝐞𝐚𝐥::𝒞, V, 1, V::F>::from_fn(|c| {
            Jet::new(point[c], [if c == i { V::F::one() } else { V::F::zero() }])
        });
        let g = f(input);
        for o in 0..G::N {
            out[o * V::N + i] = g[o][1];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffable::coords::Coords;
    use diffable::traits::{Dual, Right, Sinister, Tensor, calculus::TensorProduct};

    fn cube<V: Euclidean>(x: V) -> V {
        x.map(|x| x.powi(3))
    }

    #[test]
    fn derivative_of_cube() {
        let j = jacobian_of(cube, &Coords([2.0]));
        assert!((j[0] - 12.0).abs() < 1e-12); // d/dx x³ at 2 = 12
    }

    #[test]
    fn derivative_of_tensor_values() {
        // g: (r) -> diag(1, r²) as V* ⊗ V* — the metric-shaped output.
        type G<V> = TensorProduct<Sinister<Dual<V>>, Dual<V>>;
        fn g<V: Euclidean + Tensor<Hand = Right>>(x: V) -> G<V> {
            TensorProduct::from_fn_ij(|i, j| {
                if i == j {
                    if i == 0 { V::F::one() } else { x[0] * x[0] }
                } else {
                    V::F::zero()
                }
            })
        }
        let j = jacobian_of(g, &Coords([3.0, 0.0]));
        // out[o * N + i], o = m*N + k = 1*2 + 1, i = 0 → ∂g_11/∂x_0 = 2r = 6
        assert!((j[6] - 6.0).abs() < 1e-12);
    }
}
