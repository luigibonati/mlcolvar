__all__ = ["mse_loss","tda_loss","elbo_gaussians_loss","reduce_eigenvalues_loss","autocorrelation_loss","fisher_discriminant_loss"]

from .mse import mse_loss
from .tda_loss import tda_loss
from .eigvals import reduce_eigenvalues_loss
from .elbo import elbo_gaussians_loss
from .autocorrelation import autocorrelation_loss
from .fisher import fisher_discriminant_loss

