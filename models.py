"""models.py -- Step 4: the unified GNN model zoo.

PURPOSE
    One consistent, edge-aware interface for every architecture we compare:
        GCN, ARMA_GNN  (from ENGAGE)
        GAT, GIN, TRANSFORMER, NNConv  (from PowerGraph-Node)
    All models share:
      * the SAME forward signature  `forward(data) -> pred (N, 4)`,
      * the SAME node input contract `x = (N, 7)` and target `y = (N, 4)`,
      * the SAME physical KNOWN-VALUE RE-INJECTION at inference (`inference()`),
      * genuine EDGE-AWARENESS (every model consumes `edge_attr`).

WHY THIS STEP EXISTS (design decision D7)
    A fair architecture comparison requires every model to see the same
    information and use the same physics-aware post-processing. Two concrete
    fixes over the earlier `engage_pg` v2 attempt:
      1. GAT/GIN/Transformer there computed an edge embedding but never passed it
         to the convolutions -- so they were NOT edge-aware. Here every conv
         receives edge information (scalar `edge_weight` for GCN/ARMA;
         vector `edge_attr`/`edge_dim` for GAT/GINE/Transformer; an edge-network
         for NNConv).
      2. Only ENGAGE's models re-injected known bus quantities. Here ALL models
         inherit `inference()`, so at test time each prediction is overwritten
         with the physically KNOWN inputs per bus type (slack V/theta; PV P/V;
         PQ P/Q). This is what makes the outputs physically consistent.

HOW IT CONNECTS
    data/<CODE>/<split>/dataset.pt  (Step 3)
        -> MODELS[name](input_dim=7)  -> trained by Step 5's driver
    The registry `MODELS` is imported by the experiment drivers (Step 5), which
    simply iterate over it.

ATTRIBUTION
    GCN / ARMA_GNN adapted from ENGAGE (models.py). ARMA architecture from
    Hansen et al., "Power Flow Balancing With Decentralized Graph Neural
    Networks," IEEE T-PWRS 2023, doi:10.1109/TPWRS.2022.3195301.
    GAT/GIN/Transformer/NNConv mirror PowerGraph-Node's `model.py` layer choices.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import (
    ARMAConv,
    GATv2Conv,
    GCNConv,
    GINEConv,
    NNConv,
    TransformerConv,
)

HIDDEN = 64        # hidden width shared by all models (matches ENGAGE)
EDGE_HIDDEN = 16   # width of the vector edge embedding for attention models
N_TARGET = 4       # [p_mw, q_mvar, vm_pu, va_degree]


class BasePFGNN(nn.Module):
    """Shared skeleton: node pre-encoder, post-processing, readout, and the
    physics-aware `inference()` re-injection. Subclasses implement `_mp` (the
    message-passing stack) and build whatever edge encoder they need.
    """

    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.act_small = nn.LeakyReLU(negative_slope=0.005)

        # Node pre-encoder: raw (N, input_dim) -> (N, HIDDEN).
        self.predense1_node = nn.Linear(input_dim, HIDDEN)
        self.predense2_node = nn.Linear(HIDDEN, HIDDEN)

        # Post-processing (concatenates the raw inputs as a skip connection).
        self.postdense1 = nn.Linear(HIDDEN + input_dim, HIDDEN)
        self.postdense2 = nn.Linear(HIDDEN, HIDDEN)
        self.readout = nn.Linear(HIDDEN, N_TARGET)

    # -- message passing: implemented by each subclass -----------------------
    def _mp(self, node_emb, edge_index, edge_attr):  # pragma: no cover - abstract
        raise NotImplementedError

    # -- physics-aware known-value re-injection (shared by ALL models) --------
    def inference(self, x, pred):
        """Overwrite predicted quantities that are actually KNOWN inputs, per bus
        type. x columns: [Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree];
        pred columns: [p_mw, q_mvar, vm_pu, va_degree]."""
        with torch.no_grad():
            for node_x, node_pred in zip(x, pred):
                if node_x[0]:            # slack: V and theta are set points
                    node_pred[2] = node_x[5]  # vm_pu
                    node_pred[3] = node_x[6]  # va_degree
                elif node_x[1]:          # PV: P injection and V are known
                    node_pred[0] = node_x[3]  # p_mw
                    node_pred[2] = node_x[5]  # vm_pu
                else:                    # PQ: P and Q are known
                    node_pred[0] = node_x[3]  # p_mw
                    node_pred[1] = node_x[4]  # q_mvar
        return pred

    def forward(self, data):
        x = torch.nan_to_num(data.x, nan=0.0)               # (N, input_dim)
        edge_index = data.edge_index                        # (2, 2E)
        edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)  # (2E, 4)

        node_emb = self.act(self.predense1_node(x))
        node_emb = self.act(self.predense2_node(node_emb))

        node_emb = self._mp(node_emb, edge_index, edge_attr)  # edge-aware stack

        node_emb = torch.cat([x, node_emb], dim=1)            # skip connection
        node_emb = self.act(self.postdense1(node_emb))
        node_emb = self.act(self.postdense2(node_emb))
        pred = self.readout(node_emb)

        if not self.training:
            pred = self.inference(x, pred)
        return pred


class _ScalarEdgeMixin:
    """Builds a scalar edge weight from the 4-dim edge_attr (for GCN/ARMA)."""

    def _build_scalar_edge(self):
        self.predense1_edge = nn.Linear(4, EDGE_HIDDEN)
        self.predense2_edge = nn.Linear(EDGE_HIDDEN, 1)

    def _scalar_edge(self, edge_attr):
        e = self.act(self.predense1_edge(edge_attr))
        e = self.act_small(self.predense2_edge(e))
        return e.reshape((-1,))


class GCN(BasePFGNN, _ScalarEdgeMixin):
    """GCN with a learned scalar edge weight (ENGAGE-style)."""

    def __init__(self, input_dim: int = 7, num_layers: int = 8):
        super().__init__(input_dim)
        self._build_scalar_edge()
        self.convs = nn.ModuleList(
            [GCNConv(HIDDEN, HIDDEN, normalize=True) for _ in range(num_layers)]
        )

    def _mp(self, node_emb, edge_index, edge_attr):
        w = self._scalar_edge(edge_attr)
        for conv in self.convs:
            node_emb = self.act(conv(x=node_emb, edge_index=edge_index, edge_weight=w))
        return node_emb


class ARMA_GNN(BasePFGNN, _ScalarEdgeMixin):
    """ARMA GNN (Hansen et al. 2023), scalar edge weight."""

    def __init__(self, input_dim: int = 7, num_layers: int = 8):
        super().__init__(input_dim)
        self._build_scalar_edge()
        self.arma = ARMAConv(
            HIDDEN, HIDDEN, num_stacks=5, num_layers=num_layers,
            shared_weights=False, act=self.act, dropout=0.0, bias=True,
        )

    def _mp(self, node_emb, edge_index, edge_attr):
        w = self._scalar_edge(edge_attr)
        return self.arma(node_emb, edge_index, edge_weight=w)


class _VectorEdgeMixin:
    """Builds a vector edge embedding of size `dim` from the 4-dim edge_attr."""

    def _build_vector_edge(self, dim):
        self.edge_enc = nn.Sequential(
            nn.Linear(4, EDGE_HIDDEN), nn.LeakyReLU(0.2), nn.Linear(EDGE_HIDDEN, dim)
        )


class GAT(BasePFGNN, _VectorEdgeMixin):
    """Graph Attention (GATv2), edge features via `edge_dim`."""

    def __init__(self, input_dim: int = 7, num_layers: int = 3, heads: int = 4):
        super().__init__(input_dim)
        assert HIDDEN % heads == 0
        self._build_vector_edge(EDGE_HIDDEN)
        self.convs = nn.ModuleList([
            GATv2Conv(HIDDEN, HIDDEN // heads, heads=heads, concat=True,
                      edge_dim=EDGE_HIDDEN)
            for _ in range(num_layers)
        ])

    def _mp(self, node_emb, edge_index, edge_attr):
        e = self.edge_enc(edge_attr)
        for conv in self.convs:
            node_emb = self.act(conv(node_emb, edge_index, edge_attr=e))
        return node_emb


class GIN(BasePFGNN, _VectorEdgeMixin):
    """GINE: edge-aware GIN (edge embedding added inside the conv, so it must
    match the node hidden width)."""

    def __init__(self, input_dim: int = 7, num_layers: int = 3):
        super().__init__(input_dim)
        self._build_vector_edge(HIDDEN)
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2),
                              nn.Linear(HIDDEN, HIDDEN)),
                edge_dim=HIDDEN,
            )
            for _ in range(num_layers)
        ])

    def _mp(self, node_emb, edge_index, edge_attr):
        e = self.edge_enc(edge_attr)
        for conv in self.convs:
            node_emb = self.act(conv(node_emb, edge_index, edge_attr=e))
        return node_emb


class TRANSFORMER(BasePFGNN, _VectorEdgeMixin):
    """Graph Transformer (TransformerConv), edge features via `edge_dim`."""

    def __init__(self, input_dim: int = 7, num_layers: int = 3, heads: int = 4):
        super().__init__(input_dim)
        assert HIDDEN % heads == 0
        self._build_vector_edge(EDGE_HIDDEN)
        self.convs = nn.ModuleList([
            TransformerConv(HIDDEN, HIDDEN // heads, heads=heads, concat=True,
                            edge_dim=EDGE_HIDDEN)
            for _ in range(num_layers)
        ])

    def _mp(self, node_emb, edge_index, edge_attr):
        e = self.edge_enc(edge_attr)
        for conv in self.convs:
            node_emb = self.act(conv(node_emb, edge_index, edge_attr=e))
        return node_emb


class NN_CONV(BasePFGNN):
    """NNConv: an edge network maps the 4-dim edge_attr to a HIDDENxHIDDEN
    weight matrix used in message passing (the most edge-expressive model)."""

    def __init__(self, input_dim: int = 7, num_layers: int = 2):
        super().__init__(input_dim)
        self.convs = nn.ModuleList([
            NNConv(
                HIDDEN, HIDDEN,
                nn=nn.Sequential(nn.Linear(4, 32), nn.LeakyReLU(0.2),
                                 nn.Linear(32, HIDDEN * HIDDEN)),
                aggr="mean",
            )
            for _ in range(num_layers)
        ])

    def _mp(self, node_emb, edge_index, edge_attr):
        for conv in self.convs:
            node_emb = self.act(conv(node_emb, edge_index, edge_attr))
        return node_emb


# Registry consumed by the experiment drivers (Step 5).
MODELS = {
    "gcn": GCN,
    "arma_gnn": ARMA_GNN,
    "gat": GAT,
    "gin": GIN,
    "transformer": TRANSFORMER,
    "nnconv": NN_CONV,
}
