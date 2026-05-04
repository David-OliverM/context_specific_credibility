import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from typing import List
from packages.EinsumNet.simple_einet.layers.distributions.abstract_leaf import AbstractLeaf, dist_forward
from packages.EinsumNet.simple_einet.sampling_utils import SamplingContext
# from models.predictor import Classifier, MultiHeadClassifier
from models.predictor import Classifier

class Categorical(AbstractLeaf):
    """Categorical layer. Maps each input feature to its categorical log likelihood.

    Probabilities are modeled as unconstrained parameters and are transformed via a softmax function into [0, 1] when needed.
    """

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int, num_bins: int):
        """
        Initializes a categorical distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input data.
            num_channels (int): The number of channels in the input data.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions for each leaf.
            num_bins (int): The number of bins for the categorical distribution.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)

        # Create logits
        self.logits = nn.Parameter(torch.randn(1, num_channels, num_features, num_leaves, num_repetitions, num_bins))

    def _get_base_distribution(self, ctx: SamplingContext = None):
        # Use sigmoid to ensure, that probs are in valid range
        return dist.Categorical(logits=F.log_softmax(self.logits, dim=-1))

class ConditionalCategorical(AbstractLeaf):
    """Conditional Categorical layer. Maps each input feature to its conditional categorical log likelihood,
    where the conditioning is done via a neural network.

    Probabilities are modeled as unconstrained parameters and are transformed via a softmax function into [0, 1] when needed.
    """

    def __init__(self, num_features: int, num_channels: int, num_leaves: int, num_repetitions: int, num_bins: int, cond_fn_type, cond_fn_args):
        """
        Initializes a categorical distribution with the given parameters.

        Args:
            num_features (int): The number of features in the input data.
            num_channels (int): The number of channels in the input data.
            num_leaves (int): The number of leaves in the tree structure.
            num_repetitions (int): The number of repetitions for each leaf.
            num_bins (int): The number of bins for the categorical distribution.
        """
        super().__init__(num_features, num_channels, num_leaves, num_repetitions)
        self.logit_dim = [num_channels, num_features, num_leaves, num_repetitions, num_bins]
        # Create conditioning module
        self.cond_fn = eval(cond_fn_type)(**cond_fn_args)
        

    def forward(self, x, marginalized_scopes: List[int], x_cond=None):
        """
        Computes the forward pass of the Conditional Categorical class.

        Args:
            x (torch.Tensor): The input tensor.
            marginalized_scopes (List[int]): The marginalized scopes.
            x_cond: The context used for conditioning

        Returns:
            The output tensor.
        """
        cond_probs = self.cond_fn(x_cond)
        cond_probs = cond_probs.view([cond_probs.shape[0]]+self.logit_dim)
        d = dist.Categorical(logits=F.log_softmax(cond_probs, dim=-1))
        # Compute lls
        x = dist_forward(d, x)
        # Marginalize
        x = self._marginalize_input(x, marginalized_scopes)
        return x
    
    def _get_base_distribution(self, ctx: SamplingContext = None):
        raise NotImplementedError("This should be overriden and not called")
