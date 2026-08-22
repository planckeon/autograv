use num_traits::{NumCast, One, Zero};

use diffable::traits::{
    Atomic, BothSided, Cat, DivRing, Dual, Euclidean, Field, Left, Right, Sinister, Tensor,
    calculus::{
        Contract, Here, Jet, JetVector, OnLeft, OnRight, Reassociate, Swap, TensorProduct,
        ThroughSinister, d,
    },
};

use crate::metric::MetricField;

pub type MetricTensor<V> = TensorProduct<Sinister<Dual<V>>, Dual<V>>;
pub type Einstein<V> = MetricTensor<V>;
pub type InverseMetric<V> = TensorProduct<V, Sinister<V>>;
pub type Christoffel<V> = TensorProduct<TensorProduct<V, Dual<V>>, Dual<V>>;
pub type Riemann<V> = TensorProduct<Christoffel<V>, Dual<V>>;
pub type Ricci<V> = MetricTensor<V>;

fn commute_metric_jet<C: Cat, V: Tensor<Hand = Right, Action = BothSided>>(
    g: MetricTensor<JetVector<C, V>>,
) -> JetVector<C, MetricTensor<V>>
where
    JetVector<C, V>: Tensor<Hand = Right, Action = BothSided, F = Jet<C, V::F>>,
    Jet<C, V::F>: Field,
{
    <JetVector<C, MetricTensor<V>> as Tensor>::from_fn(|i| g[i])
}

fn commute_christoffel_jet<C: Cat, V: Tensor<Hand = Right, Action = BothSided>>(
    gamma: Christoffel<JetVector<C, V>>,
) -> JetVector<C, Christoffel<V>>
where
    JetVector<C, V>: Tensor<Hand = Right, Action = BothSided, F = Jet<C, V::F>>,
    Jet<C, V::F>: Field,
{
    <JetVector<C, Christoffel<V>> as Tensor>::from_fn(|i| gamma[i])
}

pub fn christoffel_symbols<V: Euclidean<Hand = Right, Normalization = Atomic>, M: MetricField>(
    metric: &M,
    x: &V,
) -> Christoffel<V> {
    let g: MetricTensor<V> = metric.g(x.clone());
    let g_inv: InverseMetric<V> = g.inverse();

    // this deref then clone is awkward
    let dg = (*d(|x| commute_metric_jet(metric.g(x))).at(x.clone())).clone();

    let d_lmk = dg
        .clone()
        .swap::<OnLeft<Here>>()
        .reassociate::<Right>()
        .swap::<OnRight<ThroughSinister<Here>>>()
        .reassociate::<Left>();

    let d_klm = dg
        .clone()
        .reassociate::<Right>()
        .swap::<OnRight<ThroughSinister<Here>>>()
        .reassociate::<Left>()
        .swap::<OnLeft<Here>>();

    let koszul = dg + d_lmk - d_klm;

    let gamma: Christoffel<V> = TensorProduct::pure(g_inv, Sinister(koszul))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Right>>>()
        .contract::<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>();

    let half = V::F::one().div(V::F::from_nat(2));

    gamma * half
}

pub fn torsion<V: Euclidean<Hand = Right, Normalization = Atomic>>(
    gamma: Christoffel<V>,
) -> Christoffel<V> {
    let swapped: Christoffel<V> = gamma
        .clone()
        .reassociate::<Right>()
        .swap::<OnRight<ThroughSinister<Here>>>()
        .reassociate::<Left>();

    gamma - swapped
}

pub fn torsion_tensor<V: Euclidean<Hand = Right, Normalization = Atomic>, M: MetricField>(
    metric: &M,
    x: &V,
) -> Christoffel<V> {
    torsion(christoffel_symbols(metric, x))
}

pub fn riemann_tensor<V: Euclidean<Hand = Right, Normalization = Atomic>, M: MetricField>(
    metric: &M,
    x: &V,
) -> Riemann<V> {
    let gamma: Christoffel<V> = christoffel_symbols(metric, x);

    let dgamma: Riemann<V> =
        (*d(|x| commute_christoffel_jet(christoffel_symbols(metric, &x))).at(x.clone())).clone();

    let dgamma_kl: Riemann<V> = dgamma
        .clone()
        .reassociate::<Right>()
        .swap::<OnRight<ThroughSinister<Here>>>()
        .reassociate::<Left>();

    let derivative: Riemann<V> = dgamma_kl - dgamma;

    let quadratic_raw = TensorProduct::pure(gamma.clone(), Sinister(gamma))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Right>>>()
        .contract::<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>();

    let quadratic: Riemann<V> = quadratic_raw
        .reassociate::<OnLeft<Right>>()
        .swap::<OnLeft<OnRight<ThroughSinister<Here>>>>()
        .reassociate::<OnLeft<Left>>();

    let quadratic_kl: Riemann<V> = quadratic
        .clone()
        .reassociate::<Right>()
        .swap::<OnRight<ThroughSinister<Here>>>()
        .reassociate::<Left>();

    derivative + quadratic - quadratic_kl
}

pub fn ricci_tensor<V: Euclidean<Hand = Right, Normalization = Atomic>, M: MetricField>(
    metric: &M,
    x: &V,
) -> Ricci<V> {
    riemann_tensor(metric, x)
        .reassociate::<OnLeft<Right>>()
        .swap::<OnLeft<OnRight<ThroughSinister<Here>>>>()
        .reassociate::<OnLeft<Left>>()
        .contract::<OnLeft<OnLeft<Here>>>()
}

pub fn ricci_scalar<V: Euclidean<Hand = Right, Normalization = Atomic>, M: MetricField>(
    metric: &M,
    x: &V,
) -> V::F {
    let g_inv = metric.g(x.clone()).inverse();
    let ricci = ricci_tensor(metric, x);

    let once: TensorProduct<V, Dual<V>> = TensorProduct::pure(g_inv, Sinister(ricci))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Right>>()
        .contract::<OnLeft<OnRight<ThroughSinister<Here>>>>();

    once.contract::<Here>()
}

pub fn einstein_tensor<V: Euclidean<Hand = Right, Normalization = Atomic>, M: MetricField>(
    metric: &M,
    x: &V,
) -> Einstein<V> {
    let g = metric.g(x.clone());
    let ricci = ricci_tensor(metric, x);
    let scalar = ricci_scalar(metric, x);

    let half = V::F::one() / V::F::from_nat(2);

    ricci - g * (scalar * half)
}

pub fn kretschmann_invariant<V: Euclidean<Hand = Right, Normalization = Atomic>, M: MetricField>(
    metric: &M,
    x: &V,
) -> V::F {
    type CovariantRiemann<V> =
        TensorProduct<TensorProduct<TensorProduct<Sinister<Dual<V>>, Dual<V>>, Dual<V>>, Dual<V>>;

    type Raised1<V> = TensorProduct<TensorProduct<TensorProduct<V, Dual<V>>, Dual<V>>, Dual<V>>;

    type Raised2<V> = TensorProduct<TensorProduct<TensorProduct<V, Sinister<V>>, Dual<V>>, Dual<V>>;

    type Raised3<V> =
        TensorProduct<TensorProduct<TensorProduct<V, Sinister<V>>, Sinister<V>>, Dual<V>>;

    type Raised4<V> =
        TensorProduct<TensorProduct<TensorProduct<V, Sinister<V>>, Sinister<V>>, Sinister<V>>;

    let g = metric.g(x.clone());
    let g_inv = g.clone().inverse();

    let r = riemann_tensor(metric, x);

    // R_abcd
    let r_down: CovariantRiemann<V> = TensorProduct::pure(g, Sinister(r))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Left>>>()
        .reassociate::<OnLeft<OnLeft<OnLeft<Right>>>>()
        .contract::<OnLeft<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>>();

    // ------------------------------------------------------------
    // Raise slot 0: R_abcd -> R^a_bcd
    // ------------------------------------------------------------

    let r1: Raised1<V> = TensorProduct::pure(g_inv.clone(), Sinister(r_down.clone()))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Left>>>()
        .reassociate::<OnLeft<OnLeft<OnLeft<Right>>>>()
        .contract::<OnLeft<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>>();

    // ------------------------------------------------------------
    // Raise slot 1.
    //
    // [a,b,c,d] -> [b,a,c,d]
    // raise first
    // [b,a,c,d] -> [a,b,c,d]
    // ------------------------------------------------------------

    let permuted = r1.swap::<OnLeft<OnLeft<Here>>>();

    let raised = TensorProduct::pure(g_inv.clone(), Sinister(permuted))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Left>>>()
        .reassociate::<OnLeft<OnLeft<OnLeft<Right>>>>()
        .contract::<OnLeft<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>>();

    let r2: Raised2<V> = raised.swap::<OnLeft<OnLeft<Here>>>();

    // ------------------------------------------------------------
    // Raise slot 2.
    //
    // swap b,c
    // swap a,c
    // raise first
    // undo the two swaps
    // ------------------------------------------------------------

    let permuted = r2
        .reassociate::<OnLeft<Right>>()
        .swap::<OnLeft<OnRight<ThroughSinister<Here>>>>()
        .reassociate::<OnLeft<Left>>()
        .swap::<OnLeft<OnLeft<Here>>>();

    let raised = TensorProduct::pure(g_inv.clone(), Sinister(permuted))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Left>>>()
        .reassociate::<OnLeft<OnLeft<OnLeft<Right>>>>()
        .contract::<OnLeft<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>>();

    let r3: Raised3<V> = raised
        .swap::<OnLeft<OnLeft<Here>>>()
        .reassociate::<OnLeft<Right>>()
        .swap::<OnLeft<OnRight<ThroughSinister<Here>>>>()
        .reassociate::<OnLeft<Left>>();

    // ------------------------------------------------------------
    // Raise slot 3.
    //
    // d -> front via adjacent swaps,
    // raise first,
    // undo those swaps.
    // ------------------------------------------------------------

    let permuted = r3
        // [a,b,c,d] -> [a,b,d,c]
        .reassociate::<Right>()
        .swap::<OnRight<ThroughSinister<Here>>>()
        .reassociate::<Left>()
        // [a,b,d,c] -> [a,d,b,c]
        .reassociate::<OnLeft<Right>>()
        .swap::<OnLeft<OnRight<ThroughSinister<Here>>>>()
        .reassociate::<OnLeft<Left>>()
        // [a,d,b,c] -> [d,a,b,c]
        .swap::<OnLeft<OnLeft<Here>>>();

    let raised = TensorProduct::pure(g_inv, Sinister(permuted))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Left>>>()
        .reassociate::<OnLeft<OnLeft<OnLeft<Right>>>>()
        .contract::<OnLeft<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>>();

    let r_up: Raised4<V> = raised
        // [d,a,b,c] -> [a,d,b,c]
        .swap::<OnLeft<OnLeft<Here>>>()
        // [a,d,b,c] -> [a,b,d,c]
        .reassociate::<OnLeft<Right>>()
        .swap::<OnLeft<OnRight<ThroughSinister<Here>>>>()
        .reassociate::<OnLeft<Left>>()
        // [a,b,d,c] -> [a,b,c,d]
        .reassociate::<Right>()
        .swap::<OnRight<ThroughSinister<Here>>>()
        .reassociate::<Left>();

    r_down
        .iter()
        .zip(r_up.iter())
        .fold(V::F::zero(), |sum, (down, up)| sum + *down * *up)
}

pub fn stress_energy_momentum_tensor<
    V: Euclidean<Hand = Right, Normalization = Atomic>,
    M: MetricField,
>(
    metric: &M,
    x: &V,
) -> MetricTensor<V> {
    let g = einstein_tensor(metric, x);

    let kappa = <V::F as NumCast>::from(
        (8.0 * std::f64::consts::PI * 6.67e-11) / 299_792_458.0_f64.powi(4),
    )
    .unwrap();

    // SI conversion divides by κ ≈ 2e-43, which catastrophically
    // amplifies floating-point residuals in numerically vacuum solutions.
    // Canonicalize values the scalar model already regards as zero
    // before applying the unit conversion.
    MetricTensor::<V>::from_fn(|i| {
        let value = g[i];

        if value == V::F::zero() {
            V::F::zero()
        } else {
            value / kappa
        }
    })
}
