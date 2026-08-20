use num_traits::{NumCast, One, Zero, real::Real as _};

use diffable::traits::{
    Array, BothSided, Cat, Dual, DualSinistered, Dualized, Euclidean, Field, Left, NormalizeWith,
    Real, Right, Sinister, Sinistered, Tensor, Undecorated,
    calculus::{
        Contract, Here, Jet, JetVector, OnLeft, OnRight, Reassociate, Swap, TensorProduct,
        ThroughSinister, d,
    },
    𝐑𝐞𝐚𝐥,
};

use crate::metric::MetricField;

pub fn component<T, const R: usize>(
    tensor: &T,
    index: [usize; R],
) -> T::F
where
    T: Tensor,
    T::F: Copy,
{
    let dimension = (1..=T::N)
        .find(|&i| i.pow(R as u32) == T::N)
        .expect("tensor component count is not an exact R-th power");

    let flat = index
        .into_iter()
        .fold(0, |flat, i| flat * dimension + i);

    tensor[flat]
}

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

fn inverse_metric<V>(g: MetricTensor<V>) -> InverseMetric<V>
where
    V: Tensor<Hand = Right, Action = BothSided>,
    V::F: Real,
{
    type Row<V> = <V as Tensor>::Array<<V as Tensor>::F>;

    let mut a: V::Array<Row<V>> = V::Array::from_fn(|i| V::Array::from_fn(|j| g[i * V::N + j]));

    let mut inverse: V::Array<Row<V>> = V::Array::from_fn(|i| {
        V::Array::from_fn(|j| if i == j { V::F::one() } else { V::F::zero() })
    });

    for column in 0..V::N {
        // Partial pivoting.
        let mut pivot = column;
        let mut pivot_abs = a[column][column].abs();

        for row in column + 1..V::N {
            let candidate = a[row][column].abs();

            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }

        assert!(
            !a[pivot][column].is_zero(),
            "attempted to invert a degenerate metric"
        );

        // Swap pivot row into place.
        if pivot != column {
            for j in 0..V::N {
                let tmp = a[column][j];
                a[column][j] = a[pivot][j];
                a[pivot][j] = tmp;

                let tmp = inverse[column][j];
                inverse[column][j] = inverse[pivot][j];
                inverse[pivot][j] = tmp;
            }
        }

        // Normalize the pivot row.
        let scale = a[column][column];

        for j in 0..V::N {
            a[column][j] = a[column][j] / scale;
            inverse[column][j] = inverse[column][j] / scale;
        }

        // Eliminate this column from every other row.
        for row in 0..V::N {
            if row == column {
                continue;
            }

            let scale = a[row][column];

            if scale.is_zero() {
                continue;
            }

            for j in 0..V::N {
                a[row][j] = a[row][j] - scale * a[column][j];

                inverse[row][j] = inverse[row][j] - scale * inverse[column][j];
            }
        }
    }

    InverseMetric::<V>::from_fn(|n| {
        let i = n / V::N;
        let j = n % V::N;

        inverse[i][j]
    })
}

pub fn christoffel_symbols<V, M>(metric: &M, x: &V) -> Christoffel<V>
where
    V: Euclidean<Hand = Right>
        + Copy
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, V>: Euclidean<Hand = Right, Action = BothSided, F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
    M: MetricField,
{
    type Dg<V> = TensorProduct<MetricTensor<V>, Dual<V>>;
    type RightAssociatedDg<V> =
        TensorProduct<Sinister<Dual<V>>, Sinister<TensorProduct<Sinister<Dual<V>>, Dual<V>>>>;

    let g: MetricTensor<V> = metric.g(x.clone());
    let g_inv: InverseMetric<V> = inverse_metric(g);

    // this deref then clone is awkward
    let dg: Dg<V> = (*d(|x| commute_metric_jet(metric.g(x))).at(*x)).clone();

    let d_lmk_1: RightAssociatedDg<V> = dg.clone().swap::<OnLeft<Here>>().reassociate::<Right>();

    // TODO(diffable):
    //
    //     d_lmk_2
    //         .swap::<OnRight<ThroughSinister<Here>>>()
    //
    // Once SwapKernel can descend through Sinister.
    let d_lmk_2: RightAssociatedDg<V> = TensorProduct::from_fn_ij(|i, jk| {
        let j = jk / V::N;
        let k = jk % V::N;

        d_lmk_1[(i, k * V::N + j)]
    });

    let d_lmk: Dg<V> = d_lmk_2.reassociate::<Left>();

    let d_klm_1: RightAssociatedDg<V> = dg.clone().reassociate::<Right>();

    // Same missing ThroughSinister swap.
    let d_klm_2: RightAssociatedDg<V> = TensorProduct::from_fn_ij(|i, jk| {
        let j = jk / V::N;
        let k = jk % V::N;

        d_klm_1[(i, k * V::N + j)]
    });

    let d_klm: Dg<V> = d_klm_2.reassociate::<Left>().swap::<OnLeft<Here>>();

    let koszul: Dg<V> = dg + d_lmk - d_klm;

    let gamma: Christoffel<V> = TensorProduct::pure(g_inv, Sinister(koszul))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Right>>>()
        .contract::<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>();

    let half = V::F::one() / (V::F::from_nat(2));

    gamma * half
}

pub fn torsion<V>(gamma: Christoffel<V>) -> Christoffel<V>
where
    V: Euclidean<Hand = Right>
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
{
    type RightAssociatedGamma<V> =
        TensorProduct<V, Sinister<TensorProduct<Sinister<Dual<V>>, Dual<V>>>>;

    let swapped_1: RightAssociatedGamma<V> = gamma.clone().reassociate::<Right>();

    // TODO(diffable):
    //
    // swapped_1.swap::<OnRight<ThroughSinister<Here>>>()
    let swapped_2: RightAssociatedGamma<V> = TensorProduct::from_fn_ij(|i, jk| {
        let j = jk / V::N;
        let k = jk % V::N;

        swapped_1[(i, k * V::N + j)]
    });

    let swapped: Christoffel<V> = swapped_2.reassociate::<Left>();

    gamma - swapped
}

pub fn torsion_tensor<V, M>(metric: &M, x: &V) -> Christoffel<V>
where
    V: Euclidean<Hand = Right>
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>
        + Copy,
    M: MetricField,
{
    torsion(christoffel_symbols(metric, x))
}

pub fn riemann_tensor<V, M>(metric: &M, x: &V) -> Riemann<V>
where
    V: Euclidean<Hand = Right>
        + Copy
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
    Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>: Real,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, V>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>,
        > + Copy
        + NormalizeWith<Undecorated, Normalized = JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>
        + NormalizeWith<Dualized, Normalized = Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<Sinistered, Normalized = Sinister<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>>,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
        >,
    M: MetricField,
{
    let gamma: Christoffel<V> = christoffel_symbols(metric, x);

    let dgamma: Riemann<V> =
        (*d(|x| commute_christoffel_jet(christoffel_symbols(metric, &x))).at(*x)).clone();

    let dgamma_kl: Riemann<V> = Riemann::<V>::from_fn(|n| {
        let i = n / (V::N * V::N * V::N);
        let j = (n / (V::N * V::N)) % V::N;
        let k = (n / V::N) % V::N;
        let l = n % V::N;

        dgamma[i * V::N * V::N * V::N + j * V::N * V::N + l * V::N + k]
    });

    let derivative: Riemann<V> = dgamma_kl - dgamma;

    let quadratic_raw = TensorProduct::pure(gamma.clone(), Sinister(gamma))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Right>>>()
        .contract::<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>();

    let quadratic: Riemann<V> = Riemann::<V>::from_fn(|n| {
        let i = n / (V::N * V::N * V::N);
        let j = (n / (V::N * V::N)) % V::N;
        let k = (n / V::N) % V::N;
        let l = n % V::N;

        quadratic_raw[i * V::N * V::N * V::N + k * V::N * V::N + j * V::N + l]
    });

    let quadratic_kl: Riemann<V> = Riemann::<V>::from_fn(|n| {
        let i = n / (V::N * V::N * V::N);
        let j = (n / (V::N * V::N)) % V::N;
        let k = (n / V::N) % V::N;
        let l = n % V::N;

        quadratic[i * V::N * V::N * V::N + j * V::N * V::N + l * V::N + k]
    });

    derivative + quadratic - quadratic_kl
}

pub fn ricci_tensor<V, M>(metric: &M, x: &V) -> Ricci<V>
where
    // exactly the same bounds as riemann()
    V: Euclidean<Hand = Right>
        + Copy
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
    Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>: Real,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, V>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>,
        > + Copy
        + NormalizeWith<Undecorated, Normalized = JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>
        + NormalizeWith<Dualized, Normalized = Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<Sinistered, Normalized = Sinister<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>>,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
        >,
    M: MetricField,
{
    let r = riemann_tensor(metric, x);

    // [i, j, k, l] -> [i, k, j, l]
    //
    // TODO(diffable): replace this with the appropriate structural swap.
    let contracted_order: Riemann<V> = Riemann::<V>::from_fn(|n| {
        let i = n / (V::N * V::N * V::N);
        let j = (n / (V::N * V::N)) % V::N;
        let k = (n / V::N) % V::N;
        let l = n % V::N;

        r[i * V::N * V::N * V::N + k * V::N * V::N + j * V::N + l]
    });

    contracted_order.contract::<OnLeft<OnLeft<Here>>>()
}

pub fn ricci_scalar<V, M>(metric: &M, x: &V) -> V::F
where
    // same bounds as ricci()/riemann()
    V: Euclidean<Hand = Right>
        + Copy
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, V>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>,
        > + Copy
        + NormalizeWith<Undecorated, Normalized = JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>
        + NormalizeWith<Dualized, Normalized = Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<Sinistered, Normalized = Sinister<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>>,
    Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>: Real,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
        >,
    M: MetricField,
{
    let g_inv = inverse_metric(metric.g(*x));
    let ricci = ricci_tensor(metric, x);

    let once: TensorProduct<V, Dual<V>> = TensorProduct::pure(g_inv, Sinister(ricci))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Right>>()
        .contract::<OnLeft<OnRight<ThroughSinister<Here>>>>();

    once.contract::<Here>()
}

pub fn einstein_tensor<V, M>(metric: &M, x: &V) -> Einstein<V>
where
    // same riemann bounds
    V: Euclidean<Hand = Right>
        + Copy
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, V>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>,
        > + Copy
        + NormalizeWith<Undecorated, Normalized = JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>
        + NormalizeWith<Dualized, Normalized = Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<Sinistered, Normalized = Sinister<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>>,
    Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>: Real,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
        >,
    M: MetricField,
{
    let g = metric.g(*x);
    let ricci = ricci_tensor(metric, x);
    let scalar = ricci_scalar(metric, x);

    let half = V::F::one() / V::F::from_nat(2);

    ricci - g * (scalar * half)
}

pub fn kretschmann_invariant<V, M>(metric: &M, x: &V) -> V::F
where
    // same bounds as riemann()
    V: Euclidean<Hand = Right>
        + Copy
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
    Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>: Real,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, V>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>,
        > + Copy
        + NormalizeWith<Undecorated, Normalized = JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>
        + NormalizeWith<Dualized, Normalized = Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<Sinistered, Normalized = Sinister<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>>,
    Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>: Real,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
        >,
    M: MetricField,
{
    type CovariantRiemann<V> =
        TensorProduct<TensorProduct<TensorProduct<Sinister<Dual<V>>, Dual<V>>, Dual<V>>, Dual<V>>;

    let g = metric.g(*x);
    let g_inv = inverse_metric(g.clone());

    let r: Riemann<V> = riemann_tensor(metric, x);

    let r_down: CovariantRiemann<V> = TensorProduct::pure(g, Sinister(r.clone()))
        .reassociate::<Left>()
        .reassociate::<OnLeft<Left>>()
        .reassociate::<OnLeft<OnLeft<Left>>>()
        .reassociate::<OnLeft<OnLeft<OnLeft<Right>>>>()
        .contract::<OnLeft<OnLeft<OnLeft<OnRight<ThroughSinister<Here>>>>>>();

    type Raised1<V> = TensorProduct<TensorProduct<TensorProduct<V, Dual<V>>, Dual<V>>, Dual<V>>;

    type Raised2<V> = TensorProduct<TensorProduct<TensorProduct<V, Sinister<V>>, Dual<V>>, Dual<V>>;

    type Raised3<V> =
        TensorProduct<TensorProduct<TensorProduct<V, Sinister<V>>, Sinister<V>>, Dual<V>>;

    type Raised4<V> =
        TensorProduct<TensorProduct<TensorProduct<V, Sinister<V>>, Sinister<V>>, Sinister<V>>;

    // R^{a}{}_{bcd}
    let r1: Raised1<V> = r;

    // R^{ab}{}_{cd}
    let r2: Raised2<V> = Raised2::<V>::from_fn(|n| {
        let a = n / (V::N * V::N * V::N);
        let b = (n / (V::N * V::N)) % V::N;
        let c = (n / V::N) % V::N;
        let deez = n % V::N;

        (0..V::N).fold(V::F::zero(), |sum, j| {
            sum + g_inv[b * V::N + j]
                * r1[a * V::N * V::N * V::N + j * V::N * V::N + c * V::N + deez]
        })
    });

    // R^{abc}{}_d
    let r3: Raised3<V> = Raised3::<V>::from_fn(|n| {
        let a = n / (V::N * V::N * V::N);
        let b = (n / (V::N * V::N)) % V::N;
        let c = (n / V::N) % V::N;
        let deez = n % V::N;

        (0..V::N).fold(V::F::zero(), |sum, k| {
            sum + g_inv[c * V::N + k]
                * r2[a * V::N * V::N * V::N + b * V::N * V::N + k * V::N + deez]
        })
    });

    // R^{abcd}
    let r_up: Raised4<V> = Raised4::<V>::from_fn(|n| {
        let a = n / (V::N * V::N * V::N);
        let b = (n / (V::N * V::N)) % V::N;
        let c = (n / V::N) % V::N;
        let deez = n % V::N;

        (0..V::N).fold(V::F::zero(), |sum, l| {
            sum + g_inv[deez * V::N + l]
                * r3[a * V::N * V::N * V::N + b * V::N * V::N + c * V::N + l]
        })
    });

    r_down
        .iter()
        .zip(r_up.iter())
        .fold(V::F::zero(), |sum, (down, up)| sum + *down * *up)
}

pub fn stress_energy_momentum_tensor<V, M>(metric: &M, x: &V) -> MetricTensor<V>
where
    V: Euclidean<Hand = Right>
        + Copy
        + NormalizeWith<Undecorated, Normalized = V>
        + NormalizeWith<Dualized, Normalized = Dual<V>>
        + NormalizeWith<Sinistered, Normalized = Sinister<V>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<V>>>,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, V>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>,
        > + Copy
        + NormalizeWith<Undecorated, Normalized = JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>
        + NormalizeWith<Dualized, Normalized = Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<Sinistered, Normalized = Sinister<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>
        + NormalizeWith<DualSinistered, Normalized = Sinister<Dual<JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>>>,
    Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>: Real,
    JetVector<𝐑𝐞𝐚𝐥::𝒞, JetVector<𝐑𝐞𝐚𝐥::𝒞, V>>: Euclidean<
            Array<Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>> = V::Array<
                Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
            >,
            Hand = Right,
            Action = BothSided,
            F = Jet<𝐑𝐞𝐚𝐥::𝒞, Jet<𝐑𝐞𝐚𝐥::𝒞, V::F>>,
        >,
    M: MetricField,
{
    let g = einstein_tensor(metric, x);

    let kappa = <V::F as NumCast>::from(
        (8.0 * std::f64::consts::PI * 6.67e-11)
            / 299_792_458.0_f64.powi(4),
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
