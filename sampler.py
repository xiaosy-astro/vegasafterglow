# src/vegasglow/sampler.py
import threading
import logging
from typing import Sequence, Tuple, Callable, Type, Optional
import numpy as np
import bilby
from bilby.core.prior import Uniform, LogUniform
from .types import ModelParams, Setups, ObsData, VegasMC, FitResult, ParamDef, Scale

logger = logging.getLogger(__name__)

class _log_prob:
    """
    Thread-safe log-probability callable for Bilby.
    """
    def __init__(
        self,
        data: ObsData,
        config: Setups,
        to_params: Callable[[np.ndarray], ModelParams],
        model_cls: Type[VegasMC],
    ):
        self.data = data
        self.base_cfg = config
        self.to_params = to_params
        self.model_cls = model_cls
        self._models = {}

    def __call__(self, theta: np.ndarray) -> float:
        tid = threading.get_ident()
        if tid not in self._models:
            model = self.model_cls(self.data)
            model.set(self.base_cfg)
            self._models[tid] = model
        model = self._models[tid]

        try:
            p = self.to_params(theta)
            chi2 = model.estimate_chi2(p)
            return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
        except Exception:
            return -np.inf

class BilbySampler:
    """
    High-level MCMC runner for afterglow fitting using Bilby.
    """
    def __init__(
        self,
        param_config: Tuple[Sequence[str], np.ndarray, np.ndarray, np.ndarray, Sequence[ParamDef]],
        to_params: Callable[[np.ndarray], ModelParams],
        model_cls: Type[VegasMC],
        num_workers: Optional[int] = None
    ):
        self.labels, self.init, self.pl, self.pu, self.param_defs = param_config
        self.ndim = len(self.init)
        self.to_params = to_params
        self.model_cls = model_cls
        self.num_workers = num_workers

    def run(
        self,
        data: ObsData,
        base_cfg: Setups,
        resolution: Tuple[float, float, float] = (0.3, 1, 10),
        total_steps: int = 10_000,
        burn_frac: float = 0.2,
        thin: int = 1,
        top_k: int = 10,
        sampler: str = "dynesty"  # Default to dynesty, can be changed to "emcee" or others
    ) -> FitResult:
        """
        Run MCMC using Bilby with optional resolution configuration.
        """
        # 1) Configure coarse grid
        cfg = self._make_cfg(base_cfg, *resolution)

        # 2) Define priors
        priors = {}
        for label, pl, pu, param_def in zip(self.labels, self.pl, self.pu, self.param_defs):
            if param_def.scale == Scale.LOG:
                priors[label] = LogUniform(pl, pu, name=label)
            else:
                priors[label] = Uniform(pl, pu, name=label)

        # 3) Define likelihood
        def likelihood(theta):
            return _log_prob(data, cfg, self.to_params, self.model_cls)(theta)

        bilby_likelihood = bilby.likelihood.Likelihood(priors=priors)
        bilby_likelihood.log_likelihood = lambda: likelihood(list(priors.sample(1).values())[0])

        # 4) Run Bilby sampler
        logger.info(
            "🚀 Running MCMC with %s at resolution %s for %d steps",
            sampler, resolution, total_steps
        )
        if sampler == "emcee":
            sampler_kwargs = {
                "nwalkers": 2 * self.ndim,
                "nsteps": total_steps,
                "nburn": int(burn_frac * total_steps),
                "thin": thin,
                "nthreads": self.num_workers or 1
            }
        else:  # Default to dynesty
            sampler_kwargs = {
                "nlive": 1000,  # Number of live points
                "sample": "rwalk",  # Random walk sampling
                "walks": 100,
                "maxmcmc": total_steps,
                "nact": 10,
                "pool": self.num_workers
            }

        result = bilby.run_sampler(
            likelihood=bilby_likelihood,
            priors=priors,
            sampler=sampler,
            outdir="bilby_out",
            label="vegasglow_fit",
            **sampler_kwargs
        )

        # 5) Extract samples and log probabilities
        chain = result.posterior[self.labels].to_numpy()
        logp = result.posterior["log_likelihood"].to_numpy()

        # 6) Filter bad samples (optional, similar to emcee filtering)
        chain, logp, _ = self._filter_bad_samples(chain, logp)

        # 7) Find top k fits
        flat_chain = chain
        flat_logp = logp
        sorted_idx = np.argsort(flat_logp)[::-1]
        rounded_params = np.round(flat_chain[sorted_idx], decimals=12)
        _, unique_idx = np.unique(rounded_params, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)[:top_k]
        final_idx = sorted_idx[unique_idx]
        top_k_params = flat_chain[final_idx]
        top_k_log_probs = flat_logp[final_idx]

        logger.info(
            "🎯 Found %d unique fits with log probabilities: %.2f to %.2f",
            len(top_k_params), top_k_log_probs[0], top_k_log_probs[-1]
        )

        # 8) Return FitResult
        return FitResult(
            samples=chain,
            posterior=result.posterior,
            labels=self.labels,
            top_k_params=top_k_params,
            top_k_log_probs=top_k_log_probs,
            log_evidence=result.log_evidence,
            log_evidence_err=result.log_evidence_err
        )

    def _make_cfg(self, base_cfg: Setups, phi: float, theta: float, t: float) -> Setups:
        """
        Create a shallow copy of base_cfg and override its grid resolution.
        """
        cfg = type(base_cfg)()
        for attr in dir(base_cfg):
            if not attr.startswith("_") and hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, getattr(base_cfg, attr))
                except Exception:
                    pass
        cfg.t_resol = t
        cfg.theta_resol = theta
        cfg.phi_resol = phi
        return cfg

    @staticmethod
    def _filter_bad_samples(
        chain: np.ndarray,
        logp: np.ndarray,
        threshold_mad: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove samples whose log-prob is > threshold_mad·MAD below the median.
        """
        median = np.median(logp)
        mad = np.median(np.abs(logp - median))
        cutoff = median - threshold_mad * mad
        good = logp > cutoff
        logger.info(
            "🎯 Filtered %d / %d bad samples (cutoff=%.2f)",
            np.sum(~good), len(logp), cutoff
        )
        return chain[good], logp[good], good
