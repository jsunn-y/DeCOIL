import torch
import numpy as np
import os
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal

# Define the number of objectives and input dimensions
N_OBJECTIVES = 9  # Set to any N
INPUT_DIM = 160  # 160D input space

# Custom ensemble model for high-dimensional input (160D) and N objectives
class EnsembleModel(Model):
    def __init__(self, models, num_objectives):
        super().__init__()
        self.models = models  # List of trained models
        self.num_objectives = num_objectives

    def forward(self, X):
        """Predict mean and variance from an ensemble."""
        X_np = X.detach().numpy()  # Convert tensor to numpy
        
        # Collect predictions from all models in the ensemble
        preds = np.array([model.predict(X_np) for model in self.models])  # (n_models, n_samples, N_OBJECTIVES)

        mean_pred = preds.mean(axis=0)  # Mean over ensemble members
        var_pred = preds.var(axis=0) + 1e-6  # Small value to prevent zero variance

        # Convert to BoTorch format (MultivariateNormal)
        mean_pred_torch = torch.tensor(mean_pred, dtype=torch.float32)
        cov_torch = torch.diag_embed(torch.tensor(var_pred, dtype=torch.float32))  # Variance as diagonal
        return MultivariateNormal(mean_pred_torch, cov_torch)

class SurrogateModel:
    """A dummy model that returns random predictions for 160D input (replace with actual models)."""
    def init(self, model_file):
        self.model = torch.load(model_file) 

    def predict(self, X):
        return self.model(X)

class Acquisition():
    def init(self, model_path):
        # Create an ensemble with multiple models
        model_files = os.listdir(model_path)

        ensemble_models = [SurrogateModel(model_file) for model_file in model_files]  # Replace with actual trained models
        ensemble = EnsembleModel(ensemble_models, num_objectives=N_OBJECTIVES)

        # Define reference point (worse than all Pareto-optimal points)
        ref_point = torch.zeros(N_OBJECTIVES)  # Generalized for N objectives

        # Example Pareto front (random for demonstration)
        train_Y = torch.rand((3, N_OBJECTIVES))  # 3 points, N objectives

        # Define partitioning for EHVI computation
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
        sampler = SobolQMCNormalSampler(num_samples=128)

        # Define the EHVI acquisition function
        self.ehvi = qExpectedHypervolumeImprovement(
            model=ensemble,  # Use custom ensemble model
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
    
    def __call__(self, batch):
        return self.ehvi(batch.unsqueeze(1))