//! Full flat-space spherical-coordinate verification.

use autograv::{
    SphericalPolar, as_t2, as_t3, as_t4, christoffel_symbols, einstein_tensor,
    kretschmann_invariant, ricci_scalar, ricci_tensor, riemann_tensor,
    stress_energy_momentum_tensor, torsion_tensor,
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

    let gamma = as_t3::<3>(&christoffel_symbols(&metric, &coordinates));
    let torsion = torsion_tensor(&metric, &coordinates);
    let riemann = as_t4::<3>(&riemann_tensor(&metric, &coordinates));
    let ricci = as_t2::<3>(&ricci_tensor(&metric, &coordinates));
    let einstein = as_t2::<3>(&einstein_tensor(&metric, &coordinates));
    let stress_energy = as_t2::<3>(&stress_energy_momentum_tensor(&metric, &coordinates));

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

    assert!((gamma[0][1][1] + 5.0).abs() < 1e-12);
    assert!(torsion.iter().all(|&v| v == 0.0));
    assert!(
        riemann
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .all(|&v| v == 0.0)
    );
    assert!(ricci.iter().flatten().all(|&v| v == 0.0));
    assert_eq!(ricci_scalar(&metric, &coordinates), 0.0);
    assert!(einstein.iter().flatten().all(|&v| v == 0.0));
    assert!(stress_energy.iter().flatten().all(|&v| v == 0.0));
    assert_eq!(kretschmann_invariant(&metric, &coordinates), 0.0);
}
