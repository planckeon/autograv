<p align="center">
  <img
    src="https://raw.githubusercontent.com/planckeon/autograv/main/assets/banner.png"
    alt="autograv"
    width="100%"
  />
</p>

<h1 align="center">autograv</h1>

<p align="center">
  <a href="https://crates.io/crates/autograv">
    <img src="https://img.shields.io/crates/v/autograv.svg" alt="crates.io" />
  </a>
  <a href="https://docs.rs/autograv">
    <img src="https://img.shields.io/docsrs/autograv" alt="docs.rs" />
  </a>
  <a href="https://github.com/planckeon/autograv/blob/main/LICENSE">
    <img src="https://img.shields.io/crates/l/autograv.svg" alt="license" />
  </a>
</p>

Numerical-relativity tensor calculus in Rust, built on
[`diffable`](https://crates.io/crates/diffable)'s typed tensor algebra and
forward-mode automatic differentiation.

## Installation

```bash
cargo add autograv diffable
```

`diffable` is included explicitly because coordinate values use its public
`Coords` type.

## Quick start

```rust
use autograv::{as_t3, christoffel_symbols, ricci_scalar, SphericalPolar};
use diffable::coords::Coords;

fn main() {
    let point = Coords([
        5.0,
        std::f64::consts::FRAC_PI_3,
        std::f64::consts::FRAC_PI_2,
    ]);

    let metric = SphericalPolar;
    let christoffel = as_t3::<3>(&christoffel_symbols(&metric, &point));

    // Γʳ_θθ = −r
    assert!((christoffel[0][1][1] + 5.0).abs() < 1e-12);

    // Spherical coordinates describe flat Euclidean space.
    assert_eq!(ricci_scalar(&metric, &point), 0.0);
}
```

## Implemented quantities

The crate currently computes:

- Christoffel symbols
- torsion tensor
- Riemann curvature tensor
- Ricci tensor
- Ricci scalar
- Kretschmann invariant
- Einstein tensor
- stress-energy-momentum tensor

## Built-in metrics

- `Minkowski`
- `SphericalPolar`
- `Schwarzschild`

`SphericalPolar` is the three-dimensional Euclidean metric

```text
dr² + r²dθ² + r²sin²(θ)dφ²
```

in spherical coordinates. It has nonzero coordinate connection coefficients but zero curvature.

## Coordinate and scalar model

Metric implementations are generic over their scalar type. This permits
`diffable` Taylor jets to flow through metric, connection, and curvature
calculations.

The public API uses `diffable::coords::Coords` for coordinate presentation and
nested tensor views for rank-2, rank-3, and rank-4 results.

## API documentation

- [Latest API documentation](https://docs.rs/autograv)
- [crates.io package](https://crates.io/crates/autograv)
- [Source repository](https://github.com/planckeon/autograv)

## Verification

The repository verifies:

- Minkowski flatness
- spherical-coordinate flatness
- Schwarzschild vacuum identities
- Schwarzschild Kretschmann invariant
- tensor-valued automatic differentiation
- documentation examples
- clippy and formatting cleanliness

## License

MIT. See [`LICENSE`](LICENSE).
