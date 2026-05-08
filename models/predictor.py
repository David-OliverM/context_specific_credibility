import torch


def make_encoder(
    in_dim,
    embed_dim,
    n_layers,
    n_hidden,
    activation='torch.nn.Tanh()',
    dropout=0.0
):
    layers = [
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=in_dim, out_features=n_hidden),
        eval(activation),
    ]

    for _ in range(n_layers):
        if dropout > 0.0:
            layers += [
                torch.nn.Linear(n_hidden, n_hidden),
                eval(activation),
                torch.nn.Dropout(dropout),
            ]
        else:
            layers += [
                torch.nn.Linear(n_hidden, n_hidden),
                eval(activation),
            ]

    layers += [
        torch.nn.Linear(n_hidden, embed_dim),
        eval(activation),
    ]

    return torch.nn.Sequential(*layers)


def make_head(
    embed_dim,
    out_dim,
    final_activation='torch.nn.Softmax(dim=-1)'
):
    layers = [torch.nn.Linear(embed_dim, out_dim)]

    if eval(final_activation) is not None:
        layers += [eval(final_activation)]

    return torch.nn.Sequential(*layers)


class MLPEncoder(torch.nn.Module):
    """Plain feed-forward encoder for tabular / pre-extracted features.

    Used by the Frankfurt fMRI pipeline (FC upper-triangular vectors per
    modality).  Accepts `freeze_params` to match the FusionModel kwargs
    convention; everything else is the same as `make_encoder`.
    """

    def __init__(
        self,
        in_dim,
        embed_dim=64,
        n_layers=2,
        n_hidden=128,
        activation='torch.nn.Tanh()',
        dropout=0.0,
        freeze_params=False,
    ):
        super().__init__()
        self.encoder = make_encoder(
            in_dim=in_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation=activation,
            dropout=dropout,
        )
        if freeze_params:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x, **kwargs):
        return self.encoder(x)


class Classifier(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers,
        n_hidden,
        embed_dim=64,
        activation='torch.nn.Tanh()',
        final_activation='torch.nn.Softmax(dim=-1)',
        dropout=0.0
    ):
        super().__init__()

        self.encoder = make_encoder(
            in_dim=in_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation=activation,
            dropout=dropout
        )

        self.head = make_head(
            embed_dim=embed_dim,
            out_dim=out_dim,
            final_activation=final_activation
        )

    def forward(self, x, context=None, return_embedding=False, **kwargs):
        x = torch.cat(x, dim=-1) if isinstance(x, list) else x

        z = self.encoder(x)          # [B, embed_dim]
        logits = self.head(z)        # [B, out_dim]

        if return_embedding:
            return logits, z
        return logits


