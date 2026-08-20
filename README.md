# autograv

Numerical relativity through automatic differentiation, with Python and Rust implementations.

## Implementations

### Python package

The Python implementation uses JAX for forward-mode automatic differentiation and array-based tensor algebra.

- PyPI package: [`autograv`](https://pypi.org/project/autograv/)
- Python documentation: [`README.python.md`](README.python.md)
- JAX examples: [`examples/`](examples/)

Install it with:

```bash
pip install autograv
```

or

```bash
uv pip install autograv
```

### Rust crate

The Rust implementation is a typed tensor-calculus library built on [`diffable`](https://crates.io/crates/diffable).

- crates.io package: [`autograv`](https://crates.io/crates/autograv)
- API documentation: [`docs.rs/autograv`](https://docs.rs/autograv)
- Rust documentation: [`README.rust.md`](README.rust.md)
- Rust examples: [`examples/`](examples/)

Install it with:

```bash
cargo add autograv diffable
```

## Repository layout

```text
src/autograv/       Python package
src/*.rs            Rust crate
examples/*.py       Python examples
examples/*.rs       Rust examples
tests/*.rs          Rust integration tests
paper/              Research paper and references
docs/               Project and publishing documentation
```

## License

MIT. See [`LICENSE`](LICENSE).
