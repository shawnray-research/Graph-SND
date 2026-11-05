"""Graph-SND: scalable System Neural Diversity via graph-based local
aggregation and sampling estimators."""

from graphsnd.designs import (
    Anatomy,
    anatomy,
    balanced_design,
    balanced_variance,
    degree_for_accuracy,
    design_variance,
    estimate_mean_dissimilarity,
    median_of_designs,
    metric_relvar_bound,
    round_robin_estimate,
    separable_mean,
    sparse_distinctiveness,
)
try:  # torch-dependent components; the design API above needs only numpy
    from graphsnd.batched_policies import (
        BatchedGaussianMLPPolicy,
        BatchedLinear,
        BatchedValueMLP,
        load_batched_checkpoint,
        save_batched_checkpoint,
    )
    from graphsnd.graphs import (
        bernoulli_edges,
        complete_edges,
        knn_edges,
        random_regular_edges,
        spectral_gap,
        uniform_size_edges,
    )
    from graphsnd.metrics import (
        graph_snd,
        graph_snd_from_rollouts,
        hoeffding_bound,
        ht_estimator,
        pairwise_behavioral_distance,
        pairwise_distances_on_edges,
        serfling_bound,
        snd,
        uniform_sample_estimator,
    )
    from graphsnd.wasserstein import (
        wasserstein_gaussian,
        wasserstein_gaussian_diag,
    )
except ImportError:  # pragma: no cover
    pass

__version__ = "0.1.0"

__all__ = [
    "Anatomy",
    "anatomy",
    "balanced_design",
    "balanced_variance",
    "degree_for_accuracy",
    "design_variance",
    "estimate_mean_dissimilarity",
    "median_of_designs",
    "metric_relvar_bound",
    "round_robin_estimate",
    "separable_mean",
    "sparse_distinctiveness",
    "BatchedGaussianMLPPolicy",
    "BatchedLinear",
    "BatchedValueMLP",
    "bernoulli_edges",
    "complete_edges",
    "graph_snd",
    "graph_snd_from_rollouts",
    "hoeffding_bound",
    "ht_estimator",
    "knn_edges",
    "load_batched_checkpoint",
    "pairwise_behavioral_distance",
    "pairwise_distances_on_edges",
    "random_regular_edges",
    "save_batched_checkpoint",
    "serfling_bound",
    "snd",
    "spectral_gap",
    "uniform_sample_estimator",
    "uniform_size_edges",
    "wasserstein_gaussian",
    "wasserstein_gaussian_diag",
]

# Without torch only the NumPy design API is importable.
__all__ = [name for name in __all__ if name in globals()]
