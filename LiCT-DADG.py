from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CfCEncoder(nn.Module):
    def __init__(self, input_dim: int, time_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + input_dim + time_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_1 = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_gate = nn.Linear(hidden_dim, 1)
        self.content_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_features: torch.Tensor,
        delta_t: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_nodes, _ = x.shape
        hidden = x.new_zeros(batch_size, num_nodes, self.hidden_dim)
        states = []
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1).unsqueeze(-1)
        elif delta_t.dim() == 3:
            delta_t = delta_t.unsqueeze(-1)
        delta_t = delta_t.to(dtype=x.dtype, device=x.device)
        if update_mask is None:
            update_mask = x.new_ones(batch_size, seq_len)
        if update_mask.dim() == 2:
            update_mask = update_mask.unsqueeze(-1).unsqueeze(-1)
        elif update_mask.dim() == 3:
            update_mask = update_mask.unsqueeze(-1)
        update_mask = update_mask.to(dtype=x.dtype, device=x.device)

        for step in range(seq_len):
            previous_hidden = hidden
            time_step = time_features[:, step].unsqueeze(1).expand(-1, num_nodes, -1)
            recurrent_input = torch.cat([x[:, step], mask[:, step], time_step, hidden], dim=-1)
            features = self.backbone(recurrent_input)
            h1 = torch.tanh(self.candidate_1(features))
            h2 = torch.tanh(self.candidate_2(features))
            step_delta = delta_t[:, step].expand(-1, num_nodes, -1)
            gate = torch.sigmoid(self.time_gate(features) * step_delta + self.content_gate(features))
            candidate_hidden = (1.0 - gate) * h1 + gate * h2
            step_update = update_mask[:, step].expand(-1, num_nodes, -1)
            hidden = step_update * candidate_hidden + (1.0 - step_update) * previous_hidden
            states.append(hidden)

        return torch.stack(states, dim=1)




class DelayPatternTransform(nn.Module):
    def __init__(self, hidden_dim: int, delay_dim: int, num_patterns: int):
        super().__init__()
        self.num_patterns = num_patterns
        self.query_proj = nn.Linear(hidden_dim, delay_dim)
        self.memory_proj = nn.Linear(delay_dim, delay_dim, bias=False)
        self.value_proj = nn.Linear(delay_dim, delay_dim, bias=False)
        self.prototype_bank = nn.Parameter(torch.randn(num_patterns, delay_dim) * 0.02)

    def initialize_prototypes(self, prototypes: torch.Tensor) -> None:
        if prototypes.shape != self.prototype_bank.shape:
            raise ValueError(f"Expected prototypes {tuple(self.prototype_bank.shape)}, got {tuple(prototypes.shape)}.")
        with torch.no_grad():
            self.prototype_bank.copy_(prototypes.to(device=self.prototype_bank.device, dtype=self.prototype_bank.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pattern_feature = hidden_states.mean(dim=1)
        query = self.query_proj(pattern_feature)
        memory = self.memory_proj(self.prototype_bank)
        score = torch.matmul(query, memory.t()) / (memory.shape[-1] ** 0.5)
        weight = torch.softmax(score, dim=-1)
        values = self.value_proj(self.prototype_bank)
        return torch.matmul(weight, values)


class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, order: int):
        super().__init__()
        self.order = order
        self.weights = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for _ in range(order + 1)])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        outputs = self.weights[0](x)
        propagated = x
        for hop in range(1, self.order + 1):
            if adjacency.dim() == 2:
                propagated = torch.einsum("ij,bjd->bid", adjacency, propagated)
            else:
                propagated = torch.einsum("bij,bjd->bid", adjacency, propagated)
            outputs = outputs + self.weights[hop](propagated)
        return outputs


class DynamicGraphGenerator(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        time_dim: int,
        delay_dim: int,
        embed_dim: int,
        top_k: int,
        delay_mode: str = "learned",
    ):
        super().__init__()
        self.top_k = top_k
        self.delay_mode = delay_mode
        self.filter_net = GraphConv(hidden_dim + time_dim, embed_dim * 2, order=2)
        self.base_embedding_1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.02)
        self.base_embedding_2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.delay_key_proj = nn.Linear(delay_dim, embed_dim, bias=False)
        self.delay_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        hidden: torch.Tensor,
        delay: torch.Tensor | None,
        time_feature: torch.Tensor,
        adjacency: torch.Tensor,
        fixed_delay_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = hidden.shape
        time_nodes = time_feature.unsqueeze(1).expand(-1, num_nodes, -1)
        generator_input = torch.cat([hidden, time_nodes], dim=-1)
        dynamic_filter = self.filter_net(generator_input, adjacency)
        filter_1, filter_2 = torch.chunk(dynamic_filter, 2, dim=-1)

        embedding_1 = torch.tanh(filter_1 * self.base_embedding_1)
        embedding_2 = torch.tanh(filter_2 * self.base_embedding_2)
        score_dg = F.relu(torch.tanh(torch.matmul(embedding_1, embedding_2.transpose(1, 2)) - torch.matmul(embedding_2, embedding_1.transpose(1, 2))))

        scores = score_dg
        if self.delay_mode == "learned":
            if delay is None:
                raise ValueError("Learned delay mode requires delay representations.")
            query = self.query_proj(hidden)
            key = self.key_proj(hidden) + self.delay_key_proj(delay)
            score_delay = torch.matmul(query, key.transpose(1, 2)) / (query.shape[-1] ** 0.5)
            scores = scores + self.delay_weight * score_delay
        elif self.delay_mode == "fixed":
            if fixed_delay_bias is None:
                raise ValueError("Fixed delay mode requires a fixed delay bias matrix.")
            if fixed_delay_bias.dim() == 2:
                fixed_delay_bias = fixed_delay_bias.unsqueeze(0)
            scores = scores + fixed_delay_bias.to(device=hidden.device, dtype=hidden.dtype)
        elif self.delay_mode != "none":
            raise ValueError(f"Unsupported delay mode: {self.delay_mode}")

        adjacency_dyn = torch.softmax(scores, dim=-1)
        k = min(self.top_k, adjacency_dyn.shape[-1])
        top_values, top_indices = torch.topk(adjacency_dyn, k=k, dim=-1)
        sparse = torch.zeros_like(adjacency_dyn).scatter(-1, top_indices, top_values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class PhysicalDynamicFusion(nn.Module):
    def __init__(self, hidden_dim: int, fusion_dim: int, horizon: int, output_dim: int, order: int, fusion_mode: str = "gated"):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.physical_conv = GraphConv(hidden_dim, fusion_dim, order)
        self.dynamic_conv = GraphConv(hidden_dim, fusion_dim, order)
        if fusion_mode == "linear":
            self.gate = None
            self.linear_fusion = nn.Linear(hidden_dim + fusion_dim * 2, fusion_dim)
        else:
            self.gate = nn.Linear(hidden_dim + fusion_dim * 2, fusion_dim)
            self.linear_fusion = None
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, horizon * output_dim),
        )
        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, hidden: torch.Tensor, adjacency_phy: torch.Tensor, adjacency_dyn: torch.Tensor) -> torch.Tensor:
        local_repr = self.physical_conv(hidden, adjacency_phy)
        dynamic_repr = self.dynamic_conv(hidden, adjacency_dyn)
        fusion_input = torch.cat([hidden, local_repr, dynamic_repr], dim=-1)
        if self.fusion_mode == "linear":
            if self.linear_fusion is None:
                raise ValueError("Linear fusion layer is not initialized.")
            fused = torch.relu(self.linear_fusion(fusion_input))
        elif self.fusion_mode == "gated":
            if self.gate is None:
                raise ValueError("Fusion gate is not initialized.")
            gate = torch.sigmoid(self.gate(fusion_input))
            fused = gate * local_repr + (1.0 - gate) * dynamic_repr
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        output = self.predictor(fused)
        batch_size, num_nodes, _ = output.shape
        return output.view(batch_size, num_nodes, self.horizon, self.output_dim).permute(0, 2, 1, 3)


class TrafficFlowPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        time_dim: int = 4,
        hidden_dim: int = 64,
        delay_dim: int = 32,
        embed_dim: int = 32,
        fusion_dim: int = 64,
        num_patterns: int = 16,
        top_k: int = 20,
        horizon: int = 12,
        output_dim: int = 1,
        num_nodes: int = 307,
        ablation: str = "full",
        fixed_delay_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        valid_ablations = {"full", "wo_dap", "fixed_dap", "wo_dgl", "wo_stfg"}
        if ablation not in valid_ablations:
            raise ValueError(f"Unsupported ablation '{ablation}'. Expected one of {sorted(valid_ablations)}.")
        self.ablation = ablation
        self.encoder = CfCEncoder(input_dim, time_dim, hidden_dim)
        if ablation in {"wo_dap", "wo_dgl"}:
            delay_mode = "none"
        elif ablation == "fixed_dap":
            delay_mode = "fixed"
        else:
            delay_mode = "learned"
        self.uses_learned_delay = delay_mode == "learned"
        self.delay_transform = DelayPatternTransform(hidden_dim, delay_dim, num_patterns) if self.uses_learned_delay else None
        self.dynamic_graph = DynamicGraphGenerator(num_nodes, hidden_dim, time_dim, delay_dim, embed_dim, top_k, delay_mode=delay_mode)
        fusion_mode = "linear" if ablation == "wo_stfg" else "gated"
        self.fusion = PhysicalDynamicFusion(hidden_dim, fusion_dim, horizon, output_dim, order=2, fusion_mode=fusion_mode)
        if fixed_delay_bias is not None:
            self.register_buffer("fixed_delay_bias", fixed_delay_bias.to(dtype=torch.float32))
        else:
            self.fixed_delay_bias = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_features: torch.Tensor,
        delta_t: torch.Tensor,
        adjacency: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_seq = self.encoder(x, mask, time_features, delta_t, update_mask)
        current_hidden = hidden_seq[:, -1]
        delay_repr = self.delay_transform(hidden_seq) if self.uses_learned_delay and self.delay_transform is not None else None
        if self.ablation == "wo_dgl":
            dynamic_adjacency = adjacency
        else:
            dynamic_adjacency = self.dynamic_graph(current_hidden, delay_repr, time_features[:, -1], adjacency, self.fixed_delay_bias)
        return self.fusion(current_hidden, adjacency, dynamic_adjacency)
