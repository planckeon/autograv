//! Full flat-space spherical-coordinate verification.

use autograv::{
    SphericalPolar, christoffel_symbols, component, einstein_tensor, kretschmann_invariant,
    ricci_scalar, ricci_tensor, riemann_tensor, stress_energy_momentum_tensor, torsion_tensor,
};
use diffable::coords::Coords;
use diffable::traits::Tensor;

fn main() {
    let metric = SphericalPolar;
    let coordinates = Coords([
        5.0,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
    ]);

    let gamma = &christoffel_symbols(&metric, &coordinates);
    let torsion = torsion_tensor(&metric, &coordinates);
    let riemann = &riemann_tensor(&metric, &coordinates);
    let ricci = &ricci_tensor(&metric, &coordinates);
    let einstein = &einstein_tensor(&metric, &coordinates);
    let stress_energy = &stress_energy_momentum_tensor(&metric, &coordinates);

    println!("Christoffel symbols: {gamma:?}");
    println!("Torsion tensor: {:?}", torsion.iter().collect::<Vec<_>>());
    println!("Riemann tensor: {riemann:?}");
    println!("Ricci tensor: {ricci:?}");
    println!("Ricci scalar: {}", ricci_scalar(&metric, &coordinates));
    println!("Einstein tensor: {einstein:?}");
    println!("Stress-energy tensor: {stress_energy:?}");
    println!(
        "Kretschmann invariant: {}",
        kretschmann_invariant(&metric, &coordinates)
    );

    assert!((component(gamma, [0, 1, 1]) + 5.0).abs() < 1e-12);
    assert!(torsion.iter().all(|&v| v == 0.0));
    assert!(riemann.iter().all(|&v| v == 0.0));
    assert!(ricci.iter().all(|&v| v == 0.0));
    assert_eq!(ricci_scalar(&metric, &coordinates), 0.0);
    assert!(einstein.iter().all(|&v| v == 0.0));
    assert!(stress_energy.iter().all(|&v| v == 0.0));
    assert_eq!(kretschmann_invariant(&metric, &coordinates), 0.0);
}
