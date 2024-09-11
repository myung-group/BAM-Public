from flax import linen as nn 
import e3nn_jax as e3nn
import jax.numpy as jnp
from bam_mol.model.layer import NEQUIPLayerFlax, default_radial_basis


class GraphNN (nn.Module):
    cutoff: float
    avg_num_neighbors: float
    num_species: int 
    max_ell: int = 2
    num_basis_func: int = 8
    # small GPU memory
    hidden_irreps: e3nn.Irreps = e3nn.Irreps ("32x0e+8x1o+4x2e")
    nlayers: int = 4
    features_dim : int = 128
    output_irreps: e3nn.Irreps = e3nn.Irreps("1x0e")

    @nn.compact
    def __call__ (self, Rij, data_graph):
        # Rij : (num_edges, 3)
        assert Rij.ndim == 2 and Rij.shape[1] == 3
        # iatoms ==> senders
        # jatoms ==> receivers
        iatoms = data_graph.senders
        jatoms = data_graph.receivers
        
        Rij = Rij/self.cutoff
        Rij = e3nn.IrrepsArray ("1o", Rij)
        
        # (Embedding)
        # num_embeddings = (the number of atomic species)
        species = data_graph.nodes['species']
        node_feats = nn.Embed (num_embeddings=self.num_species, 
                              features=self.features_dim) (species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[-1]}x0e", node_feats)
        
        lengths = e3nn.norm(Rij).array[:,0]
        #
        # Note that there are length == 0 owing to jraph.pad_with_graphs.
        #
        radial_embedding = jnp.where (
            (lengths == 0.0) [:,None], 0.0,
            default_radial_basis(lengths, self.num_basis_func) )

        for i in range(self.nlayers):

            layer = NEQUIPLayerFlax(
                hidden_irreps = self.hidden_irreps,
                avg_num_neighbors = self.avg_num_neighbors,
                num_species = self.num_species,
                max_ell= self.max_ell,
                )
            node_feats = \
                layer(Rij, node_feats, species, radial_embedding, iatoms, jatoms)
        
        features = e3nn.flax.Linear("16x0e")(node_feats)
        features = e3nn.flax.Linear("2x0e") (features)  # [energy, enr_var]
        features = features.array
        graph_energy = e3nn.scatter_sum(features[:, 0], nel=data_graph.n_node)
        
        return graph_energy.reshape(-1)



