from flax import linen as nn
import e3nn_jax as e3nn
import jax.numpy as jnp
import jax
from typing import Callable, Optional, Union

import flax
import haiku as hk

def default_radial_basis (r, n: int, r_max=1.0):
    """Default radial basis function.
    r: input distance,
    n: number of basis functions
    r_max: max(r)
    Polynomial envelop with p = 2
    e3nn.poly_envelope (p-1, 2) (r) = e3nn.radial.u (p, r)
    """
    return e3nn.bessel(r, n) * e3nn.poly_envelope(1, 2)(r)[:, None]
    #return e3nn.bessel(r, n) * e3nn.soft_envelope (r, r_max, value_at_origin=1.0) [:, None]

class NEQUIPLayerFlax(flax.linen.Module):
    hidden_irreps: e3nn.Irreps
    avg_num_neighbors: float
    num_species: int
    max_ell: int = 2
    #hidden_irreps: e3nn.Irreps
    even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh
    gate_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 3
    active_fn: str = 'relu'

    @flax.linen.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        node_specie: jnp.ndarray,
        radial_embedding,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ):
        return _impl(
            e3nn.flax.Linear,
            e3nn.flax.MultiLayerPerceptron,
            self,
            vectors,
            node_feats,
            node_specie,
            radial_embedding,
            senders,
            receivers,
        )



def _impl(
    Linear: Callable,
    MultiLayerPerceptron: Callable,
    self: Union[NEQUIPLayerFlax],
    vectors: e3nn.IrrepsArray,  # [n_edges, 3]
    node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
    node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
    radial_embedding,
    senders: jnp.ndarray,  # [n_edges]
    receivers: jnp.ndarray,  # [n_edges]
):
    node_feats = e3nn.as_irreps_array(node_feats)

    num_nodes = node_feats.shape[0]
    num_edges = vectors.shape[0]
    assert vectors.shape == (num_edges, 3)
    assert node_feats.shape == (num_nodes, node_feats.irreps.dim)
    assert node_specie.shape == (num_nodes,)
    assert senders.shape == (num_edges,)
    assert receivers.shape == (num_edges,)

    # we regroup the target irreps to make sure that gate activation
    # has the same irreps as the target
    hidden_irreps = e3nn.Irreps(self.hidden_irreps).regroup()

    messages = Linear(node_feats.irreps, name="linear_up")(node_feats)[senders]

    # Angular part
    messages = e3nn.concatenate(
        [
            messages.filter(hidden_irreps + "0e"),
            e3nn.tensor_product(
                messages,
                e3nn.spherical_harmonics(
                    [l for l in range(1, self.max_ell + 1)],
                    vectors,
                    normalize=True,
                    normalization="component",
                ),
                filter_ir_out=hidden_irreps + "0e",
            ),
        ]
    ).regroup()
    assert messages.shape == (num_edges, messages.irreps.dim)

    # Radial part
    with jax.ensure_compile_time_eval():
        assert abs(self.mlp_activation(0.0)) < 1e-6
    lengths = e3nn.norm(vectors).array
    mix = MultiLayerPerceptron(
        self.mlp_n_layers * (self.mlp_n_hidden,) + (messages.irreps.num_irreps,),
        self.mlp_activation,
        output_activation=False,
    )(radial_embedding) #(self.radial_basis(lengths[:, 0], self.n_radial_basis))

    # Discard 0 length edges that come from graph padding
    mix = jnp.where(lengths == 0.0, 0.0, mix)
    assert mix.shape == (num_edges, messages.irreps.num_irreps)

    # Product of radial and angular part
    messages = messages * mix
    assert messages.shape == (num_edges, messages.irreps.dim)

    # Skip connection
    irreps = hidden_irreps.filter(keep=messages.irreps)
    num_nonscalar = irreps.filter(drop="0e + 0o").num_irreps
    irreps = irreps + e3nn.Irreps(f"{num_nonscalar}x0e").simplify()

    skip = Linear(
        irreps,
        num_indexed_weights=self.num_species,
        name="skip_tp",
        force_irreps_out=True,
    )(node_specie, node_feats)

    # Message passing
    node_feats = e3nn.scatter_sum(messages, dst=receivers, output_size=num_nodes)
    node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)

    node_feats = Linear(irreps, name="linear_down")(node_feats)

    node_feats = node_feats + skip
    assert node_feats.shape == (num_nodes, node_feats.irreps.dim)

    node_feats = e3nn.gate(
        node_feats,
        even_act=self.even_activation,
        odd_act=self.odd_activation,
        even_gate_act=self.gate_activation,
    )
    if self.active_fn == 'relu':
        node_feats = jax.nn.relu(node_feats.array)
    elif self.active_fn == 'silu':
        node_feats = jax.nn.silu(node_feats.array)
    else:
        node_feats = node_feats.array  # 활성화 함수 없이 통과

    return node_feats

