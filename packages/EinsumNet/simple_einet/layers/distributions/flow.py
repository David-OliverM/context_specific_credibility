from typing import Tuple

import torch
from torch import distributions as dist
from torch import nn

from packages.EinsumNet.simple_einet.layers.distributions.abstract_leaf import AbstractLeaf
from packages.EinsumNet.simple_einet.sampling_utils import SamplingContext
from packages.EinsumNet.simple_einet.type_checks import check_valid


class NormalizingFlow(AbstractLeaf):
    """Gaussian layer. Maps each input feature to its gaussian log likelihood."""

    def __init__(
        self,
        num_features: int,
        num_channels: int,
        num_leaves: int,
        num_repetitions: int,
    ):
        """
        Initializes a Normal distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input tensor.
            num_channels (int): The number of channels in the input tensor.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions of the tree structure.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create gaussian means and stds
        self.means = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions))
        self.log_stds = nn.Parameter(torch.rand(1, num_channels, num_features, num_leaves, num_repetitions))

        self.transforms =  torch.nn.ModuleList([])
        self.initialize_flow()
    
    def update_flow_dist(self):
        self.base_dist  = self.base_dist_type(*self.base_dist_params)
        self.flow_dist = dist.ConditionalTransformedDistribution(self.base_dist,self.transforms)
        
    def initialize_flow(self):
        self.base_dist  = self.base_dist_type(*self.base_dist_params)
        self.num_stats = 0
        for p in self.transforms.parameters():
            self.num_stats += np.prod(list(p.size()))
        self.generative_flows = list(itertools.chain(*zip(self.transforms)))
        self.normalizing_flows = self.generative_flows[::-1] # normalizing direction (x-->z)
        self.flow_dist = dist.ConditionalTransformedDistribution(self.base_dist,self.transforms)
        self.flow_dist.clear_cache()
        
    def _get_base_distribution(self, ctx: SamplingContext = None):
        return dist.Normal(loc=self.means, scale=self.log_stds.exp())

