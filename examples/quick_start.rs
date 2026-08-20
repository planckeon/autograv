//! Minimal autograv usage: evaluate a metric and derive its connection.

use autograv::{SphericalPolar, christoffel_symbols, component, ricci_scalar};
use diffable::coords::Coords;

fn main() {
    let metric = SphericalPolar;
    let coordinates = Coords([
        5.0,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
    ]);
    let christoffel = christoffel_symbols(&metric, &coordinates);

    println!("Γ^r_θθ = {}", component(&christoffel, [0, 1, 1]));
    println!("Ricci scalar = {}", ricci_scalar(&metric, &coordinates));

    assert!((component(&christoffel, [0, 1, 1]) + 5.0).abs() < 1e-12);
    assert_eq!(ricci_scalar(&metric, &coordinates), 0.0);
}
