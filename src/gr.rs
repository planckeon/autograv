//! General-relativity index algebra: connection, curvature, Einstein field
//! equations. The math mirrors the Python reference implementation exactly
//! (verified formula-by-formula); only the differentiation entry point is
//! ours (see `crate::ad`).

use num_traits::{Zero, real::Real};

use diffable::coords::Coords;
use diffable::traits::{
    Dual, Euclidean, Right, Tensor,
    calculus::{JetRegion, TensorProduct},
    ι, 𝐑𝐞𝐚𝐥,
};

use crate::ad::jacobian_of;
use crate::metric::{MetricField, ScalarConst};
use crate::tensor::{
    T2, T3, T4, close_to_zero, close_to_zero_scalar, flatten_3, flatten_4, from_flat_2,
    from_flat_3, from_flat_4, invert, permute3, permute4, to_flat,
};

/// Γ^j_kl — mixed `(1, 2)` connection tensor.
pub type Christoffel<V> = TensorProduct<TensorProduct<V, Dual<V>>, Dual<V>>;
/// R^j_klm — mixed `(1, 3)` curvature tensor.
pub type Riemann<V> = TensorProduct<Christoffel<V>, Dual<V>>;
/// R_kl — covariant `(0, 2)` tensor (same shape as the metric).
pub type Ricci<V> = crate::metric::MetricTensor<V>;

/// Γ^j_kl = ½ g^jm (∂_l g_mk + ∂_k g_lm − ∂_m g_kl), evaluated at `x`.
///
/// Generic over the point presentation `V` so the same code runs at real
/// points (`f64`) and at jet points (nested `JetVector`) — the latter is what
/// lets `riemann_tensor` differentiate the connection.
pub(crate) fn christoffel_generic<V, const N: usize, M>(metric: &M, x: V) -> Christoffel<V>
where
    V: Euclidean + Tensor<Hand = Right> + Copy,
    V::F: Real + ScalarConst + ι<C: JetRegion<𝐑𝐞𝐚𝐥::𝒞>>,
    M: MetricField<N>,
{
    // g_ij and its inverse at this point.
    let g = metric.g(x);
    let g_arr: T2<V::F, N> = from_flat_2(&to_flat(&g));
    let g_inv = invert(&g_arr);

    // ∂g_ab/∂x^c via diffable jets: D[a][b][c].
    let d_flat = jacobian_of(|v| metric.g(v), &x);
    let d: T3<V::F, N> = from_flat_3(&d_flat);

    // S[m][k][l] = ∂_l g_mk + ∂_k g_lm − ∂_m g_kl
    let d_mlk = permute3(&d, [2, 0, 1]); // dst[m][k][l] = d[l][m][k]
    let d_klm = permute3(&d, [1, 2, 0]); // dst[m][k][l] = d[k][l][m]

    let mut gamma = [[[V::F::zero(); N]; N]; N];
    // 0.5 * Σ_m g^jm · S[m][k][l]
    let half = V::F::from_const(0.5);
    for j in 0..N {
        for k in 0..N {
            for l in 0..N {
                let mut s = V::F::zero();
                for m in 0..N {
                    s = s + g_inv[j][m] * (d[m][k][l] + d_mlk[m][k][l] - d_klm[m][k][l]);
                }
                gamma[j][k][l] = s * half;
            }
        }
    }

    Christoffel::<V>::from_fn(|i| flatten_3(&gamma)[i])
}

/// Affine connection coefficients (Christoffel symbols of the second kind).
pub fn christoffel_symbols<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> Christoffel<Coords<f64, N>> {
    close_to_zero(christoffel_generic(metric, *x))
}

/// T^j_kl = Γ^j_kl − Γ^j_lk. Identically zero for a Levi-Civita connection —
/// a verification of the machinery, matching the Python reference.
pub fn torsion_tensor<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> Christoffel<Coords<f64, N>> {
    let g = christoffel_generic(metric, *x);
    let flat = to_flat(&g);
    let arr: T3<f64, N> = from_flat_3(&flat);
    let swapped = permute3(&arr, [0, 2, 1]); // dst[j][k][l] = arr[j][l][k]
    let mut out = [[[0.0; N]; N]; N];
    for j in 0..N {
        for k in 0..N {
            for l in 0..N {
                out[j][k][l] = arr[j][k][l] - swapped[j][k][l];
            }
        }
    }
    close_to_zero(Christoffel::<Coords<f64, N>>::from_fn(|i| {
        flatten_3(&out)[i]
    }))
}

/// R^j_klm = ∂_m Γ^j_kl − ∂_l Γ^j_km + Γ^j_rm Γ^r_kl − Γ^j_rl Γ^r_km.
pub(crate) fn riemann_raw<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> T4<f64, N> {
    let gamma_flat = to_flat(&christoffel_generic(metric, *x));
    let gamma: T3<f64, N> = from_flat_3(&gamma_flat);

    // ∂_m Γ^j_kl via jets through the generic connection pipeline.
    let d_flat = jacobian_of(|v| christoffel_generic(metric, v), x);
    let dg: T4<f64, N> = from_flat_4(&d_flat);
    let dg_swap = permute4(&dg, [0, 1, 3, 2]); // dst[j][k][l][m] = dg[j][k][m][l]

    let mut r = [[[[0.0; N]; N]; N]; N];
    for j in 0..N {
        for k in 0..N {
            for l in 0..N {
                for m in 0..N {
                    let t3: f64 = (0..N).map(|s| gamma[j][s][m] * gamma[s][k][l]).sum();
                    let t4: f64 = (0..N).map(|s| gamma[j][s][l] * gamma[s][k][m]).sum();
                    r[j][k][l][m] = dg[j][k][l][m] - dg_swap[j][k][l][m] + t3 - t4;
                }
            }
        }
    }
    r
}

/// Riemann curvature tensor.
pub fn riemann_tensor<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> Riemann<Coords<f64, N>> {
    let r = riemann_raw(metric, x);
    close_to_zero(Riemann::<Coords<f64, N>>::from_fn(|i| flatten_4(&r)[i]))
}

/// R_kl = Σ_j R^j_klj.
pub fn ricci_tensor<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> Ricci<Coords<f64, N>> {
    let r = riemann_raw(metric, x);
    let mut out = [[0.0; N]; N];
    for k in 0..N {
        for l in 0..N {
            out[k][l] = (0..N).map(|j| r[j][k][l][j]).sum();
        }
    }
    close_to_zero(Ricci::<Coords<f64, N>>::from_fn(|i| out[i / N][i % N]))
}

/// R = g^kl R_kl.
pub fn ricci_scalar<const N: usize, M: MetricField<N>>(metric: &M, x: &Coords<f64, N>) -> f64 {
    let g_arr: T2<f64, N> = from_flat_2(&to_flat(&metric.g(*x)));
    let g_inv = invert(&g_arr);
    let ricci = to_flat(&ricci_tensor(metric, x));
    let mut s = 0.0;
    for k in 0..N {
        for l in 0..N {
            s += g_inv[k][l] * ricci[k * N + l];
        }
    }
    close_to_zero_scalar(s)
}

/// K = R^ijkl R_ijkl — the Kretschmann invariant, coordinate-independent
/// singularity detector. Port of the Python einsum chain:
///   upper[i][p][q][r] = Σ_jkl g^pj g^qk g^rl R[i][j][k][l]
///   lower[p][j][k][l] = Σ_i  g_pi R[i][j][k][l]
///   K = Σ_ijkl upper[i][j][k][l] · lower[i][j][k][l]
pub fn kretschmann_invariant<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> f64 {
    let r = riemann_raw(metric, x);
    let g_arr: T2<f64, N> = from_flat_2(&to_flat(&metric.g(*x)));
    let g_inv = invert(&g_arr);

    let mut upper = [[[[0.0; N]; N]; N]; N];
    let mut lower = [[[[0.0; N]; N]; N]; N];
    for i in 0..N {
        for p in 0..N {
            for q in 0..N {
                for rr in 0..N {
                    let mut s = 0.0;
                    for j in 0..N {
                        for k in 0..N {
                            for l in 0..N {
                                s += g_inv[p][j] * g_inv[q][k] * g_inv[rr][l] * r[i][j][k][l];
                            }
                        }
                    }
                    upper[i][p][q][rr] = s;
                }
            }
        }
    }
    for p in 0..N {
        for j in 0..N {
            for k in 0..N {
                for l in 0..N {
                    lower[p][j][k][l] = (0..N).map(|i| g_arr[p][i] * r[i][j][k][l]).sum();
                }
            }
        }
    }
    let k = (0..N)
        .flat_map(|i| {
            (0..N).flat_map(move |j| {
                (0..N).flat_map(move |k| (0..N).map(move |l| upper[i][j][k][l] * lower[i][j][k][l]))
            })
        })
        .sum();
    close_to_zero_scalar(k)
}

/// G_ij = R_ij − ½ g_ij R — the left-hand side of the Einstein field equations.
pub fn einstein_tensor<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> Ricci<Coords<f64, N>> {
    let rt = to_flat(&ricci_tensor(metric, x));
    let rs = ricci_scalar(metric, x);
    let g_arr: T2<f64, N> = from_flat_2(&to_flat(&metric.g(*x)));
    let mut out = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = rt[i * N + j] - 0.5 * g_arr[i][j] * rs;
        }
    }
    close_to_zero(Ricci::<Coords<f64, N>>::from_fn(|k| out[k / N][k % N]))
}

/// T_ij = G_ij / κ with κ = 8πG/c⁴ — mass-energy content from the field
/// equations. Constants match the Python reference exactly.
pub fn stress_energy_momentum_tensor<const N: usize, M: MetricField<N>>(
    metric: &M,
    x: &Coords<f64, N>,
) -> Ricci<Coords<f64, N>> {
    let g = to_flat(&einstein_tensor(metric, x));
    let kappa = (8.0 * std::f64::consts::PI * 6.67e-11) / 299_792_458.0_f64.powi(4);
    let out: Vec<f64> = g.iter().map(|v| v / kappa).collect();
    close_to_zero(Ricci::<Coords<f64, N>>::from_fn(|i| out[i]))
}
