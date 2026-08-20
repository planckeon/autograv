//! Schwarzschild vacuum and Kretschmann verification.

use autograv::{
    Schwarzchild, as_t2, as_t3, as_t4, christoffel_symbols, einstein_tensor, kretschmann_invariant,
    ricci_scalar, ricci_tensor, riemann_tensor, stress_energy_momentum_tensor, torsion_tensor,
};
use diffable::coords::Coords;
use diffable::traits::Tensor;

fn main() {
    let gravitational_constant = 6.67e-11_f64;
    let speed_of_light = 299_792_458.0_f64;
    let mass = 4.297e6_f64 * 1.989e30_f64;
    let schwarzschild_radius = 2.0 * gravitational_constant * mass / speed_of_light.powi(2);
    let metric = Schwarzchild {
        rs: schwarzschild_radius,
        c: speed_of_light,
    };
    let coordinates = Coords([
        3600.0,
        3000.0,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
    ]);

    let gamma = as_t3::<4>(&christoffel_symbols(&metric, &coordinates));
    let riemann = as_t4::<4>(&riemann_tensor(&metric, &coordinates));
    let ricci = as_t2::<4>(&ricci_tensor(&metric, &coordinates));
    let einstein = as_t2::<4>(&einstein_tensor(&metric, &coordinates));
    let stress_energy = as_t2::<4>(&stress_energy_momentum_tensor(&metric, &coordinates));
    let scalar = ricci_scalar(&metric, &coordinates);
    let kretschmann = kretschmann_invariant(&metric, &coordinates);
    let analytic = 48.0 * gravitational_constant.powi(2) * mass.powi(2)
        / (speed_of_light.powi(4) * coordinates[1].powi(6));

    println!("Mass: {mass:.3e} kg");
    println!("Schwarzschild radius: {schwarzschild_radius:.3e} m");
    println!("Γ^j_kl: {gamma:?}");
    let curvature_scale = riemann
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let relative_roundoff = 1e-12 * curvature_scale;
    let kappa = (8.0 * std::f64::consts::PI * gravitational_constant) / speed_of_light.powi(4);

    println!("Riemann tensor max component: {curvature_scale}");
    println!("Ricci tensor: {ricci:?}");
    println!("Ricci scalar: {scalar}");
    println!("Einstein tensor: {einstein:?}");
    println!("Stress-energy tensor: {stress_energy:?}");
    println!("Kretschmann: {kretschmann} (analytic: {analytic})");

    // In these SI coordinates g₀₀ is ~10¹⁷ while angular components are ~10⁷;
    // Ricci's vacuum cancellation therefore has normal f64 relative roundoff.
    assert!(
        torsion_tensor(&metric, &coordinates)
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(
        ricci
            .iter()
            .flatten()
            .all(|&v| v.abs() <= relative_roundoff)
    );
    assert_eq!(scalar, 0.0);
    assert!(
        einstein
            .iter()
            .flatten()
            .all(|&v| v.abs() <= relative_roundoff)
    );
    assert!(
        stress_energy
            .iter()
            .flatten()
            .all(|&v| v.abs() <= relative_roundoff / kappa)
    );
    assert!((kretschmann - analytic).abs() < 1e-10 * analytic);
}
