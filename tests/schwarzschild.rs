//! Schwarzschild vacuum checks. Dimensionless parameters (rs = 2, c = 1) keep
//! the analytical values amenable, K = 12·rs²/r⁶, Ricci/Einstein/SEM = 0.

use autograv::{
    Schwarzschild, christoffel_symbols, einstein_tensor, kretschmann_invariant, ricci_scalar,
    ricci_tensor, stress_energy_momentum_tensor, torsion_tensor,
};
use diffable::traits::Tensor;
use diffable::{coords::Coords, epsilon_metric::R64};
use num_traits::zero;

fn metric() -> Schwarzschild {
    Schwarzschild { rs: 2.0, c: 1.0 }
}

fn coords() -> Coords<R64, 4> {
    Coords([0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0].map(R64))
}

#[test]
fn vacuum_quantities_vanish() {
    let (m, x) = (metric(), coords());
    assert!(torsion_tensor(&m, &x).iter().all(|&v| v == zero()));
    assert!(ricci_tensor(&m, &x).iter().all(|&v| v == zero()));
    assert!(ricci_scalar(&m, &x) == zero());
    assert!(einstein_tensor(&m, &x).iter().all(|&v| v == zero()));
    println!("{:?}", stress_energy_momentum_tensor(&m, &x));
    assert!(
        stress_energy_momentum_tensor(&m, &x)
            .iter()
            .all(|&v| v == zero())
    );
}

#[test]
fn kretschmann_matches_analytic() {
    let (m, x) = (metric(), coords());
    let computed = kretschmann_invariant(&m, &x);
    let analytic = R64(12.0 * 2.0_f64.powi(2) / 10.0_f64.powi(6)); // 12 rs² / r⁶
    assert_eq!(
        computed, analytic,
        "computed {computed} vs analytic {analytic}"
    );
}

#[test]
fn christoffels_nonzero_away_from_flatness() {
    let (m, x) = (metric(), coords());
    let gamma = &christoffel_symbols(&m, &x);
    // Γ^t_rt = Γ^t_tr = (rs/r²)/2 / (1 − rs/r)·c²  ... just check it's not zero
    assert!(gamma.iter().any(|&v| v != zero()));
}
