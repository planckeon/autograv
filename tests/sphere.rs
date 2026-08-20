//! Spherical-coordinate checks. This is the 3D Euclidean metric
//! `dr² + r²dθ² + r²sin²θdφ²`, so the nonzero connection is a coordinate
//! effect while every curvature tensor vanishes.

use autograv::{
    SphericalPolar, as_t3, christoffel_symbols, einstein_tensor, kretschmann_invariant,
    ricci_scalar, ricci_tensor, riemann_tensor, stress_energy_momentum_tensor, torsion_tensor,
};
use diffable::coords::Coords;
use diffable::traits::Tensor;

fn point() -> Coords<f64, 3> {
    Coords([
        5.0,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
    ])
}

#[test]
fn sphere_christoffels() {
    let m = SphericalPolar;
    let x = point();
    let gamma = as_t3::<3>(&christoffel_symbols(&m, &x));

    // Standard values for ds² = dr² + r²dθ² + r²sin²θ dφ²:
    // Γ^r_θθ = −r          Γ^θ_rθ = Γ^θ_θr = 1/r
    // Γ^θ_φφ = −sinθcosθ   Γ^φ_rφ = Γ^φ_φr = 1/r   Γ^φ_θφ = Γ^φ_φθ = cotθ
    assert!((gamma[0][1][1] - (-5.0)).abs() < 1e-12);
    assert!((gamma[1][0][1] - 0.2).abs() < 1e-12);
    assert!((gamma[1][1][0] - 0.2).abs() < 1e-12);
    let sc = std::f64::consts::FRAC_PI_3.sin() * std::f64::consts::FRAC_PI_3.cos();
    assert!((gamma[1][2][2] - (-sc)).abs() < 1e-12);
    assert!((gamma[2][0][2] - 0.2).abs() < 1e-12);
    let cot = std::f64::consts::FRAC_PI_3.cos() / std::f64::consts::FRAC_PI_3.sin();
    assert!((gamma[2][1][2] - cot).abs() < 1e-12);
    assert!((gamma[2][2][1] - cot).abs() < 1e-12);
}

#[test]
fn sphere_torsion_vanishes() {
    let m = SphericalPolar;
    let x = point();
    assert!(torsion_tensor(&m, &x).iter().all(|&v| v == 0.0));
}

#[test]
fn spherical_coordinates_are_flat() {
    let m = SphericalPolar;
    let x = point();
    assert!(riemann_tensor(&m, &x).iter().all(|&v| v == 0.0));
    assert!(ricci_tensor(&m, &x).iter().all(|&v| v == 0.0));
    assert_eq!(ricci_scalar(&m, &x), 0.0);
    assert!(einstein_tensor(&m, &x).iter().all(|&v| v == 0.0));
    assert!(
        stress_energy_momentum_tensor(&m, &x)
            .iter()
            .all(|&v| v == 0.0)
    );
    assert_eq!(kretschmann_invariant(&m, &x), 0.0);
}
