//! Minkowski spacetime: flat, so every curvature quantity vanishes.
use autograv::{
    Minkowski, christoffel_symbols, einstein_tensor, kretschmann_invariant, ricci_scalar,
    ricci_tensor, riemann_tensor, stress_energy_momentum_tensor,
};
use diffable::coords::Coords;
use diffable::traits::Tensor;

#[test]
fn flat_space_all_zero() {
    let m = Minkowski;
    let x = Coords([10.0, -2.0, 3.5, 0.25]);
    assert!(christoffel_symbols(&m, &x).iter().all(|&v| v == 0.0));
    assert!(riemann_tensor(&m, &x).iter().all(|&v| v == 0.0));
    assert!(ricci_tensor(&m, &x).iter().all(|&v| v == 0.0));
    assert_eq!(ricci_scalar(&m, &x), 0.0);
    assert_eq!(kretschmann_invariant(&m, &x), 0.0);
    assert!(einstein_tensor(&m, &x).iter().all(|&v| v == 0.0));
    assert!(
        stress_energy_momentum_tensor(&m, &x)
            .iter()
            .all(|&v| v == 0.0)
    );
}
