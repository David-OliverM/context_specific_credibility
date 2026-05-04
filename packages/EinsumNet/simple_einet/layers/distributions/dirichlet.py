import torch
from torch import distributions as dist
from torch import nn

from packages.EinsumNet.simple_einet.layers.distributions.abstract_leaf import AbstractLeaf


class Dirichlet(AbstractLeaf):
    """Dirichlet layer. Maps each input feature to its Dirichlet log likelihood.
    """

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int, num_classes: int):
        """
        Initializes a Dirichlet distribution with the given parameters.

        Args:
            num_classes (int): The number of classes in the input data.
            num_features (int): The number of features in the input data.
            num_channels (int): The number of channels in the input data.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions for each leaf.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create bernoulli parameters
        self.probs = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions, num_classes))

    def _get_base_distribution(self):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Dirichlet(concentration=torch.sigmoid(self.probs))
