//! Flat index algebra over tensor coordinates.
//!
//! ponytail: nested `[F; N]` arrays instead of an ndarray dependency — ranks
//! here are ≤ 4 and dims ≤ 4, and nested arrays keep const-generic array
//! lengths legal on stable (no `generic_const_exprs` until out of nightly).
//! TODO: maybe evaluate `ndarray` for the tensor algebra? or an alternate crate like `nalgebra`
//! is this ponytail the best design?

use num_traits::{NumCast, Zero, real::Real};

use diffable::traits::Tensor;

pub const TOLERANCE: f64 = 1e-8;

pub type T1<F, const N: usize> = [F; N];
pub type T2<F, const N: usize> = [[F; N]; N];
pub type T3<F, const N: usize> = [[[F; N]; N]; N];
pub type T4<F, const N: usize> = [[[[F; N]; N]; N]; N];

/// Flat, output-major coordinates of any diffable tensor (`Tensor::iter` order)
pub fn to_flat<T: Tensor>(t: &T) -> Vec<T::F> {
    t.iter().copied().collect()
}

/// View a tensor's coordinates as nested arrays (rank 2), for indexed access.
pub fn as_t2<const N: usize>(t: &impl Tensor<F = f64>) -> T2<f64, N> {
    from_flat_2(&to_flat(t))
}

/// View a tensor's coordinates as nested arrays (rank 3), for indexed access.
pub fn as_t3<const N: usize>(t: &impl Tensor<F = f64>) -> T3<f64, N> {
    from_flat_3(&to_flat(t))
}

/// View a tensor's coordinates as nested arrays (rank 4), for indexed access.
pub fn as_t4<const N: usize>(t: &impl Tensor<F = f64>) -> T4<f64, N> {
    from_flat_4(&to_flat(t))
}

pub fn from_flat_2<F: Copy, const N: usize>(v: &[F]) -> T2<F, N> {
    core::array::from_fn(|i| core::array::from_fn(|j| v[i * N + j]))
}

pub fn from_flat_3<F: Copy, const N: usize>(v: &[F]) -> T3<F, N> {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| core::array::from_fn(|k| v[(i * N + j) * N + k]))
    })
}

pub fn from_flat_4<F: Copy, const N: usize>(v: &[F]) -> T4<F, N> {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            core::array::from_fn(|k| core::array::from_fn(|l| v[((i * N + j) * N + k) * N + l]))
        })
    })
}

pub fn flatten_3<F: Copy, const N: usize>(t: &T3<F, N>) -> Vec<F> {
    let mut v = Vec::with_capacity(N * N * N);
    for plane in t {
        for row in plane {
            v.extend(row.iter().copied());
        }
    }
    v
}

pub fn flatten_4<F: Copy, const N: usize>(t: &T4<F, N>) -> Vec<F> {
    let mut v = Vec::with_capacity(N * N * N * N);
    for block in t {
        for plane in block {
            for row in plane {
                v.extend(row.iter().copied());
            }
        }
    }
    v
}

/// `dst[a][b][c] = src[p₀→a][p₁→b][p₂→c]` where `pₓ` selects which destination
/// index feeds that source slot (0 → a, 1 → b, 2 → c).
pub fn permute3<F: Copy, const N: usize>(src: &T3<F, N>, [p0, p1, p2]: [usize; 3]) -> T3<F, N> {
    let sel = |p: usize, a: usize, b: usize, c: usize| match p {
        0 => a,
        1 => b,
        _ => c,
    };
    core::array::from_fn(|a| {
        core::array::from_fn(|b| {
            core::array::from_fn(|c| src[sel(p0, a, b, c)][sel(p1, a, b, c)][sel(p2, a, b, c)])
        })
    })
}

pub fn permute4<F: Copy, const N: usize>(src: &T4<F, N>, [p0, p1, p2, p3]: [usize; 4]) -> T4<F, N> {
    let sel = |p: usize, a: usize, b: usize, c: usize, d: usize| match p {
        0 => a,
        1 => b,
        2 => c,
        _ => d,
    };
    core::array::from_fn(|a| {
        core::array::from_fn(|b| {
            core::array::from_fn(|c| {
                core::array::from_fn(|d| {
                    src[sel(p0, a, b, c, d)][sel(p1, a, b, c, d)][sel(p2, a, b, c, d)]
                        [sel(p3, a, b, c, d)]
                })
            })
        })
    })
}

/// Gauss–Jordan inverse with partial pivoting, generic over a `Real` field:
/// `f64` at evaluation time, diffable's `Jet` during differentiation.
///
/// Panics on a singular matrix (the zero primal trips diffable's `NonZero`
/// division invariant) — the API is deliberately total, like the Python
/// reference
/// TODO: but maybe there's a more idiomatic way to do it in Rust?
pub fn invert<F: Real, const N: usize>(m: &T2<F, N>) -> T2<F, N> {
    let mut a = *m;
    let mut inv = core::array::from_fn(|i| {
        core::array::from_fn(|j| if i == j { F::one() } else { F::zero() })
    });

    // Normalize each row before elimination. This is an elementary row
    // operation applied to both halves of [A | I], so it preserves A⁻¹ while
    // avoiding catastrophic cancellation when metric components have very
    // different physical scales (for example g₀₀ ~ c² and g_rr ~ 1).
    for row in 0..N {
        let mut scale = F::zero();
        for value in a[row] {
            let magnitude = value.abs();
            if magnitude > scale {
                scale = magnitude;
            }
        }
        for value in &mut a[row] {
            *value = *value / scale;
        }
        for value in &mut inv[row] {
            *value = *value / scale;
        }
    }

    for col in 0..N {
        let mut piv = col;
        for row in (col + 1)..N {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for j in 0..N {
            a[col][j] = a[col][j] / d;
            inv[col][j] = inv[col][j] / d;
        }
        for row in 0..N {
            if row == col {
                continue;
            }
            let f = a[row][col];
            for j in 0..N {
                a[row][j] = a[row][j] - f * a[col][j];
                inv[row][j] = inv[row][j] - f * inv[col][j];
            }
        }
    }

    inv
}

/// Python's `close_to_zero` decorator: round coordinates within `TOLERANCE` to
/// zero. Applied at each public GR boundary, not inside the AD path.
pub fn close_to_zero<T: Tensor>(t: T) -> T
where
    T::F: Real,
{
    let tol = <T::F as NumCast>::from(TOLERANCE).unwrap();
    T::from_fn(|i| {
        let v = t[i];
        if v.abs() < tol { T::F::zero() } else { v }
    })
}

/// Scalar variant of `close_to_zero`
pub fn close_to_zero_scalar(x: f64) -> f64 {
    if x.abs() < TOLERANCE { 0.0 } else { x }
}

// TODO: maybe need more tests here
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_recovers_inverse() {
        let m: T2<f64, 3> = [[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 25.0]];
        let inv = invert(&m);
        for (i, row) in m.iter().enumerate() {
            for (j, column) in inv.iter().enumerate() {
                let expect = if i == j { 1.0 } else { 0.0 };
                let got: f64 = row.iter().zip(column.iter()).map(|(a, b)| a * b).sum();
                assert!((got - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn permute3_cyclic() {
        // dst[a][b][c] = src[c][a][b]
        let src: T3<f64, 2> = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let dst = permute3(&src, [2, 0, 1]);
        assert_eq!(dst[0][0][0], src[0][0][0]);
        assert_eq!(dst[1][0][0], src[0][1][0]);
        assert_eq!(dst[0][1][0], src[0][0][1]);
    }
}
