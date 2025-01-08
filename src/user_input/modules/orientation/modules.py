import torch
from torch import nn
from torch import distributions as D
import torch.nn.functional as F
from torch.distributions.mixture_same_family import MixtureSameFamily
from modules.orientation.sparsemax import Sparsemax  # Replace with nn.Softmax(dim=-1) if needed


###############################################################################
#  ProjectionHead
###############################################################################
class ProjectionHead(nn.Module):
    """
    A small feedforward block used in multi-head projections.
    """
    def __init__(self, in_dim, out_dim, nlayers=1):
        super().__init__()
        self.head = nn.Sequential()
        for i in range(nlayers - 1):
            self.head.add_module(f"linear_{i}", nn.Linear(in_dim, in_dim))
            self.head.add_module(f"relu_{i}", nn.ReLU())
        self.head.add_module("linear_final", nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.head(x)


###############################################################################
#  MultiHeadLinearProjection
###############################################################################
class MultiHeadLinearProjection(nn.Module):
    """
    Generates multiple outputs in parallel. Used by HyperNet to produce sets
    of [W1, b1, W2, b2] for each predicted dimension.
    """
    def __init__(self, output_size, in_dim, nlayers=1):
        super().__init__()
        self.linears = nn.ModuleList()
        for out_dim in output_size:
            self.linears.append(ProjectionHead(in_dim, out_dim, nlayers))

    def forward(self, features):
        """
        features: [batch_size, in_dim]
        returns a list of Tensors, each shaped [batch_size, out_dim].
        """
        out = []
        for head in self.linears:
            out.append(head(features))
        return out


###############################################################################
#  HyperNet
###############################################################################
class HyperNet(nn.Module):
    """
    Dynamically generates the parameters for 'JointNetTemplate'.

    Suppose total_dims = 6 (for 6 joint angles). For each dimension i in [0..5]:
      - W1: shape (i+1) * hidden_layer_size
      - b1: shape hidden_layer_size
      - W2: shape hidden_layer_size * output_dim
      - b2: shape output_dim
    """
    def __init__(self, cfg):
        super().__init__()
        # We have 6 predicted output dimensions (e.g. 6 joints).
        self.total_dims = cfg.num_joints  # e.g. 6
        self.embedding_dim = cfg.embedding_dim
        self.hidden_layer_sizes = [cfg.hypernet_hidden_size] * cfg.hypernet_num_hidden_layers

        # Feedforward layers (hypernet backbone) with 'hypernet_input_dim' = 6
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(cfg.hypernet_input_dim, self.hidden_layer_sizes[0]))
        for i in range(len(self.hidden_layer_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_layer_sizes[i],
                                         self.hidden_layer_sizes[i+1]))

        # Output embedding
        self.out = nn.Linear(self.hidden_layer_sizes[-1], self.embedding_dim)

        # Build parameter list for all 6 dimensions
        # Each dimension has 4 parameter blocks: [W1, b1, W2, b2].
        num_parameters_lst = []
        for i in range(self.total_dims):
            num_parameters_lst += [
                cfg.jointnet_hidden_size * (i+1),           # W1
                cfg.jointnet_hidden_size,                   # b1
                cfg.jointnet_hidden_size * cfg.jointnet_output_dim,  # W2
                cfg.jointnet_output_dim                     # b2
            ]

        # Multi-head to produce all these parameters at once
        self.projection = MultiHeadLinearProjection(
            num_parameters_lst, in_dim=self.embedding_dim, nlayers=1
        )
        self.nonlinearity = torch.relu

    def forward(self, x):
        """
        x: [batch_size, hypernet_input_dim=6],
           e.g. [TCP_X, TCP_Y, TCP_Z, Orientation_X, Orientation_Y, Orientation_Z]
        """
        for layer in self.layers:
            x = self.nonlinearity(layer(x))
        embedding = self.out(x)  # [batch_size, embedding_dim]
        weights = self.projection(embedding)
        # returns a list of Tensors of length = 4 * self.total_dims
        return weights


###############################################################################
#  JointNetTemplate
###############################################################################
class JointNetTemplate(nn.Module):
    """
    Consumes the i-th dimension's inputs plus the 4 parameter Tensors [W1, b1, W2, b2],
    outputs the distribution parameters for the chain dimension i
    (e.g. mu, log_sigma, mixture weights, etc.).
    """
    def __init__(self, cfg):
        super().__init__()
        self.hidden_layer_size = cfg.jointnet_hidden_size
        # Typically: output_dim = 2 * num_gaussians (+ num_gaussians if mixture weights).
        # E.g. if num_gaussians=1 => output_dim=2, else => output_dim=3 * num_gaussians, etc.
        self.output_dim = cfg.jointnet_output_dim

    def forward(self, inp, weights):
        """
        inp: [batch_size, i+1]
        weights: a list of 4 Tensors => [W1, b1, W2, b2]
          W1 => [batch_size, (i+1) * hidden_layer_size]
          b1 => [batch_size, hidden_layer_size]
          W2 => [batch_size, hidden_layer_size * output_dim]
          b2 => [batch_size, output_dim]
        """
        W1, b1, W2, b2 = weights
        batch_size, in_features = inp.shape

        # 1) First linear layer
        if in_features == 1:
            # If there's only one input dimension, do elementwise multiplication
            out = inp * W1 + b1
        else:
            W1_reshaped = W1.reshape(batch_size, in_features, self.hidden_layer_size)
            out = torch.bmm(inp.unsqueeze(1), W1_reshaped).squeeze(1) + b1

        out = F.relu(out)

        # 2) Second linear layer
        W2_reshaped = W2.reshape(batch_size, self.hidden_layer_size, self.output_dim)
        out = torch.bmm(out.unsqueeze(1), W2_reshaped).squeeze(1) + b2

        return out


###############################################################################
#  MainNet
###############################################################################
class MainNet(nn.Module):
    """
    Produces 6 distributions in a chain-of-dependencies approach.
    Dimension i sees the previous i outputs as input.

    That is, dimension 0 sees x[:, :1],
              dimension 1 sees x[:, :2],
              ...
              dimension 5 sees x[:, :6].
    """
    def __init__(self, cfg):
        super().__init__()
        self.total_dims = cfg.num_joints  # e.g. 6
        self.num_gaussians = cfg.num_gaussians
        self.joint_template = JointNetTemplate(cfg)

    def forward(self, x, weights):
        """
        x: [batch_size, total_dims=6]
           These are the "chain inputs". Typically you might feed the (TCP_X, TCP_Y, ...).
           BUT note that this code is set up so that dimension i sees x[:, : i+1].
           If you truly want to chain, ensure your x is shaped [batch_size, 6].
        weights: list of length 4*total_dims from the HyperNet

        Returns:
          - distributions: list[MixtureSameFamily] of length = 6
          - selection: the mixture weights for each dimension
        """
        # shape => [batch_size, 6, 1]
        x = x.unsqueeze(2)

        out = []
        for i in range(self.total_dims):
            dim_weights = weights[4*i : 4*i + 4]
            # For dimension i, input is x[:, : i+1, :]
            out_i = self.joint_template(x[:, : (i+1)].squeeze(2), dim_weights)
            out.append(out_i)

        # Build mixture-of-Gaussians distributions
        distributions = []
        selection = []
        for i in range(self.total_dims):
            # out[i] has shape [batch_size, output_dim], e.g. 2*g + g => 3*g if num_gaussians>1
            if self.num_gaussians == 1:
                # Single Gaussian => no mixture weights needed
                sel_w = torch.ones_like(out[i][:, :1])
            else:
                # If mixture => parse mixture weights from the last portion
                sel_w = Sparsemax()(out[i][:, 2*self.num_gaussians:])

            selection.append(sel_w)
            mix = D.Categorical(sel_w)

            # Means & log_stds from the first 2*g portion
            means = out[i][:, :self.num_gaussians].unsqueeze(-1)
            log_stds = out[i][:, self.num_gaussians: 2*self.num_gaussians].unsqueeze(-1)

            comp = D.Independent(D.Normal(loc=means, scale=log_stds.exp() + 1e-7), 1)
            gmm = MixtureSameFamily(mix, comp)
            distributions.append(gmm)

        return distributions, selection

    def validate(self, x, weights, lower, upper):
        """
        Samples each dimension [0..5] in sequence, then clips to [lower[i], upper[i]].

        x: [batch_size, 6], typically you might start with x[:,0]=some known val,
           but it's up to you how you feed chain inputs.
        lower, upper: each must be length=6, containing the valid joint range for each dimension.

        Returns:
          - samples: list of [batch_size, 1] Tensors (the sample for each dimension)
          - distributions: the GMM for each dimension
          - means, variances, selection: for debugging
        """
        samples = []
        distributions = []
        means = []
        variances = []
        selection = []

        batch_size = x.shape[0]
        # total_dims=6
        total_dims = self.total_dims

        # Start with just x[:, 0] => shape [batch_size, 1]
        curr_input = x[:, 0].unsqueeze(1)

        for i in range(total_dims):
            dim_weights = weights[4*i : 4*i + 4]
            out_i = self.joint_template(curr_input, dim_weights)

            if self.num_gaussians == 1:
                sel_w = torch.ones(batch_size, 1, device=x.device)
            else:
                sel_w = Sparsemax()(out_i[:, 2*self.num_gaussians:])

            # Means + log_stds
            mu = out_i[:, :self.num_gaussians].unsqueeze(2)
            sigma = out_i[:, self.num_gaussians : 2*self.num_gaussians].exp().unsqueeze(2)

            selection.append(sel_w)
            means.append(mu)
            variances.append(sigma)

            mix = D.Categorical(sel_w)
            comp = D.Independent(D.Normal(loc=mu, scale=sigma + 1e-7), 1)
            dist = MixtureSameFamily(mix, comp)
            distributions.append(dist)

            # Sample => shape [batch_size, 1]
            sample = dist.sample()
            sample = sample.clip(lower[i], upper[i])

            # For the chain: dimension i+1 sees all previously sampled dims
            curr_input = torch.cat((curr_input, sample), dim=1)
            samples.append(sample)

        return samples, distributions, means, variances, selection
