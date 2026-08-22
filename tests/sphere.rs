//! Spherical-coordinate checks. This is the 3D Euclidean metric
//! `dr² + r²dθ² + r²sin²θdφ²`, so the nonzero connection is a coordinate
//! effect while every curvature tensor vanishes.

use autograv::{
    SphericalPolar, christoffel_symbols, einstein_tensor, kretschmann_invariant, ricci_scalar,
    ricci_tensor, riemann_tensor, stress_energy_momentum_tensor, torsion_tensor,
};
use diffable::coords::Coords;
use diffable::epsilon_metric::R64;
use diffable::traits::Tensor;
use num_traits::zero;

fn point() -> Coords<R64, 3> {
    Coords(
        [
            5.0,
            std::f64::consts::FRAC_PI_3,
            std::f64::consts::FRAC_PI_2,
        ]
        .map(R64),
    )
}

#[test]
fn sphere_christoffels() {
    let m = SphericalPolar;
    let x = point();
    let gamma = &christoffel_symbols(&m, &x);

    let idx = |i: usize, j: usize, k: usize| i * 9 + j * 3 + k;

    // Standard values for ds² = dr² + r²dθ² + r²sin²θ dφ²:
    // Γ^r_θθ = −r          Γ^θ_rθ = Γ^θ_θr = 1/r
    // Γ^θ_φφ = −sinθcosθ   Γ^φ_rφ = Γ^φ_φr = 1/r   Γ^φ_θφ = Γ^φ_φθ = cotθ
    assert_eq!(gamma[idx(0, 1, 1)], R64(-5.0));
    assert_eq!(gamma[idx(1, 0, 1)], R64(0.2));
    assert_eq!(gamma[idx(1, 1, 0)], R64(0.2));
    let sc = std::f64::consts::FRAC_PI_3.sin() * std::f64::consts::FRAC_PI_3.cos();
    assert_eq!(gamma[idx(1, 2, 2)], R64(-sc));
    assert_eq!(gamma[idx(2, 0, 2)], R64(0.2));
    let cot = std::f64::consts::FRAC_PI_3.cos() / std::f64::consts::FRAC_PI_3.sin();
    assert_eq!(gamma[idx(2, 1, 2)], R64(cot));
    assert_eq!(gamma[idx(2, 2, 1)], R64(cot));
}

#[test]
fn sphere_torsion_vanishes() {
    let m = SphericalPolar;
    let x = point();
    assert!(torsion_tensor(&m, &x).iter().all(|&v| v == zero()));
}

#[test]
fn spherical_coordinates_are_flat() {
    let m = SphericalPolar;
    let x = point();
    assert!(riemann_tensor(&m, &x).iter().all(|&v| v == zero()));
    assert!(ricci_tensor(&m, &x).iter().all(|&v| v == zero()));
    assert_eq!(ricci_scalar(&m, &x), zero());
    assert!(einstein_tensor(&m, &x).iter().all(|&v| v == zero()));
    assert!(
        stress_energy_momentum_tensor(&m, &x)
            .iter()
            .all(|&v| v == zero())
    );
    assert_eq!(kretschmann_invariant(&m, &x), zero());
}
