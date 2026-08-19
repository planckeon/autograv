//! # autograv
//!
//! Numerical-relativity tensor calculus built on top of
//! [`diffable`](https://docs.rs/diffable)'s typed tensor algebra and
//! forward-mode automatic differentiation.
//!
//! The crate evaluates a user-supplied [`MetricField`] and computes:
//!
//! - Christoffel symbols and torsion;
//! - Riemann curvature;
//! - Ricci tensor and scalar;
//! - Kretschmann invariant;
//! - Einstein tensor and stress-energy-momentum tensor.
//!
//! Metric functions are generic over their scalar type, allowing diffable's
//! Taylor jets to flow through the metric and connection formulas. The public
//! `jacobian_of` bridge in [`ad`] performs the seed/extract operation required
//! for tensor-valued maps, which diffable 0.5.0 does not yet represent through
//! its `d(f)` blanket implementation.
//!
//! ## Quick start
//!
//! ```
//! use autograv::{as_t3, christoffel_symbols, ricci_scalar, SphericalPolar};
//! use diffable::coords::Coords;
//!
//! let metric = SphericalPolar;
//! let point = Coords([5.0, std::f64::consts::FRAC_PI_3, std::f64::consts::FRAC_PI_2]);
//! let gamma = as_t3::<3>(&christoffel_symbols(&metric, &point));
//!
//! assert!((gamma[0][1][1] + 5.0).abs() < 1e-12); // Γ^r_θθ = −r
//! assert_eq!(ricci_scalar(&metric, &point), 0.0); // flat R³ in spherical coordinates
//! ```
//!
//! ## Coordinate conventions
//!
//! `Coords<f64, N>` is used as the coordinate presentation. The metric values,
//! not the `Coords` type parameter, define the physical signature; for example,
//! [`Minkowski`] supplies `(-,+,+,+)` and [`Schwarzschild`] supplies its
//! Lorentzian diagonal components.
//!
//! ## Numerical convention
//!
//! Public tensor results are rounded componentwise to zero when their absolute
//! value is below [`TOLERANCE`], matching the Python reference's
//! `close_to_zero` decorator. A singular metric panics during inversion; callers
//! must evaluate only at nonsingular coordinate points.
//! TODO: a better API surface than panicking?

// diffable's public API uses mathematical Unicode identifiers (𝐑𝐞𝐚𝐥, ι, 𝒞);
// we must reference them, so silence the confusability lints crate-wide.
#![allow(uncommon_codepoints, confusable_idents, mixed_script_confusables)]

pub mod ad;
pub mod gr;
pub mod metric;
pub mod tensor;

pub use gr::{
    Christoffel, Ricci, Riemann, christoffel_symbols, einstein_tensor, kretschmann_invariant,
    ricci_scalar, ricci_tensor, riemann_tensor, stress_energy_momentum_tensor, torsion_tensor,
};
pub use metric::{MetricField, MetricTensor, Minkowski, ScalarConst, Schwarzchild, SphericalPolar};
pub use tensor::{
    T1, T2, T3, T4, TOLERANCE, as_t2, as_t3, as_t4, close_to_zero, close_to_zero_scalar,
};

/*
 * TODOs:
 * - need to get rid of old docs after scrapping them for parts
 * - need to generalize library docs and metadata to be JAX when its Python, but not for the Rust bits,
 *   Rust bits is obviously done in diffable
 *      - update README.md for this
 * - why do we need setup-git.ps1? do we need setup-git.ps1? or is it a one time legacy artifact that can be cleaned?
 * - need to publish autograv to crates.io with proper README, release/tag on gh, proper versioning on crates.io, and to make sure docs show up on docs.rs
 * - too many mentions of "Python reference" in Rust version's source files, need to make standalone one in idiomatic Rust,
 *   not port design patterns from Python version as-is which might not be the best way to do something
 * - need to update typst paper and revise it with latest updates at the end at once after Rust work is done
 * - can we improve ergonomics of the API exposed by the lib by furnishing convenient macros/proc macros to the consumer?
 */

// ===============================================

// DEFERRED for MUCH LATER:
// - for future work and further scope, check einstein fields paper (https://arxiv.org/abs/2507.11589, https://github.com/AndreiB137/EinFields) and try to enhance lib
