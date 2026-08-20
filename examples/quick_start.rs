//! Minimal autograv usage: evaluate a metric and derive its connection.

use autograv::{SphericalPolar, as_t3, christoffel_symbols, ricci_scalar};
use diffable::coords::Coords;

fn main() {
    let metric = SphericalPolar;
    let coordinates = Coords([
        5.0,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
    ]);
    let christoffel = as_t3::<3>(&christoffel_symbols(&metric, &coordinates));

    println!("Γ^r_θθ = {}", christoffel[0][1][1]);
    println!("Ricci scalar = {}", ricci_scalar(&metric, &coordinates));

    assert!((christoffel[0][1][1] + 5.0).abs() < 1e-12);
    assert_eq!(ricci_scalar(&metric, &coordinates), 0.0);
}
