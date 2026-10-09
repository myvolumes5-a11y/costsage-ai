"""
CostSage AI - Differentiable Neuro-Fuzzy Multi-Task Estimator
File: src/models/neural_net.py

Architecture Overview:
1. FuzzyMembershipLayer:
   - Differentiable Gaussian membership functions with trainable centers (c)
     and log-sigmas (spreads).
   - Replaces manual fuzzy partitioning with end-to-end backpropagation.
2. Shared Representation Backbone:
   - Aggregates fuzzy activations, applies batch normalization, non-linear ReLU,
     and dropout for regularization.
3. Multi-Task Output Heads:
   - Head 1 (Regression): Predicts project effort in Person-Months (PM).
   - Head 2 (Classification): Predicts risk probability across 3 classes 
     (Low=0, Moderate=1, High=2).
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# SECTION 1: LEARNABLE GAUSSIAN FUZZY MEMBERSHIP LAYER
# ============================================================================
class FuzzyMembershipLayer(nn.Module):
    """
    Computes smooth Gaussian fuzzy membership grades for each input dimension:
        μ_ij(x_i) = exp( - (x_i - c_ij)^2 / (2 * σ_ij^2) )
    
    Both centers (c) and spreads (σ) are PyTorch nn.Parameters optimized
    via gradient descent during model training.
    """

    def __init__(self, in_features: int = 6, num_fuzzy_sets: int = 3):
        """
        Args:
            in_features: Number of incoming inputs (kloc, cplx, pcap, acap, tool, ai_tier).
            num_fuzzy_sets: Number of linguistic terms per feature (e.g., Low, Medium, High).
        """
        super().__init__()
        self.in_features = in_features
        self.num_fuzzy_sets = num_fuzzy_sets

        # Initialize membership centers uniformly across standard normalized feature bounds [-1.5, 1.5]
        # Shape: (in_features, num_fuzzy_sets)
        init_centers = torch.linspace(-1.5, 1.5, num_fuzzy_sets).repeat(in_features, 1)
        self.centers = nn.Parameter(init_centers)

        # Store log(sigma) instead of raw sigma to guarantee standard deviation stays strictly positive
        # via exp(log_sigmas) in forward pass
        self.log_sigmas = nn.Parameter(torch.zeros(in_features, num_fuzzy_sets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation:
            x shape: (batch_size, in_features)
            output shape: (batch_size, in_features * num_fuzzy_sets)
        """
        # Expand x dimensions for element-wise broadcasting: (batch_size, in_features, 1)
        x_expanded = x.unsqueeze(-1)

        # Expand centers and sigmas for batch broadcasting: (1, in_features, num_fuzzy_sets)
        centers_exp = self.centers.unsqueeze(0)
        sigmas_exp = torch.exp(self.log_sigmas).unsqueeze(0) + 1e-5  # Add epsilon to prevent division by zero

        # Compute Gaussian fuzzy membership grades: exp( -0.5 * ((x - c) / sigma)^2 )
        membership_grades = torch.exp(-0.5 * ((x_expanded - centers_exp) / sigmas_exp) ** 2)

        # Flatten all membership features for dense backbone processing
        batch_size = x.size(0)
        return membership_grades.view(batch_size, -1)


# ============================================================================
# SECTION 2: MULTI-TASK NEURO-FUZZY MODEL ARCHITECTURE
# ============================================================================
class NeuroFuzzyCostSageNet(nn.Module):
    """
    End-to-end multi-task network combining fuzzy membership representation
    with shared representation learning and task-specific prediction heads.
    """

    def __init__(self, in_features: int = 6, num_fuzzy_sets: int = 3, hidden_dim: int = 32):
        super().__init__()
        # Total fuzzy representation features entering the backbone
        fuzzy_output_dim = in_features * num_fuzzy_sets

        # 1. Trainable Fuzzy Layer
        self.fuzzy_layer = FuzzyMembershipLayer(in_features, num_fuzzy_sets)

        # 2. Shared Latent Representation Backbone
        # Learns cross-feature interactions between fuzzy memberships
        self.shared_backbone = nn.Sequential(
            nn.Linear(fuzzy_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Stabilizes gradient updates on small benchmark batches
            nn.ReLU(),
            nn.Dropout(0.15),           # Regularization to prevent overfitting on tabular data
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 3. Effort Regression Head (Predicts continuous Person-Months)
        self.effort_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # 4. Risk Classification Head (Predicts 3 risk logits: Low, Moderate, High)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Outputs:
            pred_effort: Tensor of shape (batch_size, 1), non-negative person-months.
            risk_logits: Tensor of shape (batch_size, 3), unnormalized class logits.
        """
        fuzzy_features = self.fuzzy_layer(x)
        shared_embedding = self.shared_backbone(fuzzy_features)

        # ReLU ensures software effort cannot mathematically predict negative months
        pred_effort = torch.relu(self.effort_head(shared_embedding))
        risk_logits = self.risk_head(shared_embedding)

        return pred_effort, risk_logits


# ============================================================================
# SECTION 3: MULTI-TASK TRAINING LOOP
# ============================================================================
def train_neuro_fuzzy_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_effort_train: np.ndarray,
    y_risk_train: np.ndarray,
    epochs: int = 150,
    lr: float = 0.01,
    risk_loss_weight: float = 0.5
) -> nn.Module:
    """
    Trains the multi-task model using joint backpropagation:
        Total Loss = MSE_Loss(Effort) + (risk_loss_weight * CrossEntropy_Loss(Risk))
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Loss criteria for multi-task learning
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    # Convert training inputs to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_effort_tensor = torch.tensor(y_effort_train, dtype=torch.float32)
    y_risk_tensor = torch.tensor(y_risk_train, dtype=torch.long)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass through fuzzy layer, backbone, and both heads
        pred_effort, pred_risk_logits = model(X_tensor)

        # Compute dual-head multi-task loss
        loss_effort = mse_criterion(pred_effort, y_effort_tensor)
        loss_risk = ce_criterion(pred_risk_logits, y_risk_tensor)
        total_loss = loss_effort + (risk_loss_weight * loss_risk)

        # Backpropagation and parameter optimization
        total_loss.backward()
        optimizer.step()

    return model


# ============================================================================
# SECTION 4: AI CODING TOOL TRADE-OFF SIMULATOR
# ============================================================================
def simulate_ai_tier_impact(
    baseline_effort_pm: float,
    team_size: int = 4,
    monthly_salary_per_dev: float = 8500.0
) -> Dict[str, Dict[str, Any]]:
    """
    Calculates delivery metrics across all three AI tiers:
    1. Traditional (No AI)
    2. Free / Open-Source AI (Autocomplete / Local Models)
    3. Premium AI (Cursor / Copilot @ $30/mo)

    Incorporates empirical net speedup and verification review drag.
    """
    tier_configs = {
        "No AI (Traditional)": {
            "speedup": 1.00,
            "tool_cost_monthly": 0.0,
            "review_drag": 0.00
        },
        "Free AI (Local Models)": {
            "speedup": 1.20,
            "tool_cost_monthly": 0.0,
            "review_drag": 0.12  # +12% time checking bugs/hallucinations
        },
        "Premium AI (Cursor / Copilot)": {
            "speedup": 1.45,
            "tool_cost_monthly": 30.0,  # $30 per dev per month subscription
            "review_drag": 0.05  # +5% inspection buffer
        }
    }

    results = {}
    baseline_payroll = 0.0

    for name, cfg in tier_configs.items():
        # Net speedup accounting for verification review drag
        net_speedup = cfg["speedup"] / (1.0 + cfg["review_drag"])
        adjusted_effort = baseline_effort_pm / net_speedup
        duration_months = adjusted_effort / max(1, team_size)

        payroll_cost = adjusted_effort * monthly_salary_per_dev
        tooling_cost = duration_months * team_size * cfg["tool_cost_monthly"]
        total_project_cost = payroll_cost + tooling_cost

        if name == "No AI (Traditional)":
            baseline_payroll = total_project_cost

        results[name] = {
            "adjusted_effort_pm": round(adjusted_effort, 1),
            "duration_months": round(duration_months, 1),
            "payroll_cost": round(payroll_cost, 0),
            "tooling_cost": round(tooling_cost, 0),
            "total_cost": round(total_project_cost, 0),
            "net_savings_vs_traditional": round(baseline_payroll - total_project_cost, 0)
        }

    return results


if __name__ == "__main__":
    # Smoke test for tensor dimensions and forward pass
    test_model = NeuroFuzzyCostSageNet(in_features=6, num_fuzzy_sets=3)
    sample_input = torch.randn(4, 6)
    eff, risk = test_model(sample_input)
    print("Smoke Test Passed:")
    print(f"Effort output shape: {eff.shape}")
    print(f"Risk logits shape:   {risk.shape}")
