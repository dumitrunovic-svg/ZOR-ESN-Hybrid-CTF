"""
Echo State Network (ESN) for chaotic time series forecasting.

Hyperparameters obtained through evolutionary hyperparameter search.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class ESN:
    """Leaky Echo State Network with ridge regression readout.

    Parameters
    ----------
    reservoir_size : int
        Number of reservoir neurons.
    spectral_radius : float
        Spectral radius of the reservoir weight matrix.
    input_scaling : float
        Scaling factor for input weights.
    leaking_rate : float
        Leaking rate α ∈ (0, 1]. α=1 → standard ESN, α→0 → slow integrator.
    ridge_alpha : float
        L2 regularization for ridge regression readout.
    seed : int
        Random seed for reproducibility.
    washout : int
        Number of initial steps to discard (reservoir warm-up).
    """

    def __init__(
        self,
        reservoir_size: int = 500,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leaking_rate: float = 1.0,
        ridge_alpha: float = 1e-6,
        seed: int = 42,
        washout: int = 100,
    ):
        self.N = int(reservoir_size)
        self.rho = float(spectral_radius)
        self.sigma = float(input_scaling)
        self.alpha = float(leaking_rate)
        self.ridge = float(ridge_alpha)
        self.washout = int(washout)
        self.rng = np.random.default_rng(seed)

        self.Win: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.Wout: np.ndarray | None = None
        self.d_in: int = 0
        self.d_out: int = 0

    def _build_reservoir(self, d_in: int) -> None:
        """Build input and reservoir weight matrices."""
        self.d_in = d_in
        # Input weights: uniform in [-sigma, sigma]
        self.Win = self.rng.uniform(-self.sigma, self.sigma, (self.N, d_in))

        # Sparse reservoir: ~10% connectivity
        density = 0.1
        W = sparse.random(
            self.N, self.N, density=density,
            format="csr", random_state=int(self.rng.integers(2**31)),
            data_rvs=lambda n: self.rng.uniform(-1, 1, n),
        ).toarray()

        # Scale to target spectral radius
        eigvals = np.linalg.eigvals(W)
        current_rho = np.max(np.abs(eigvals))
        if current_rho > 1e-10:
            W *= self.rho / current_rho
        self.W = W

    def _run_reservoir(self, data: np.ndarray) -> np.ndarray:
        """Drive reservoir with input data.

        Parameters
        ----------
        data : ndarray, shape (T, d)

        Returns
        -------
        states : ndarray, shape (T, N)
        """
        T, d = data.shape
        x = np.zeros(self.N)
        states = np.zeros((T, self.N))

        for t in range(T):
            u = data[t]
            pre = self.W @ x + self.Win @ u
            x = (1.0 - self.alpha) * x + self.alpha * np.tanh(pre)
            states[t] = x

        return states

    def fit(self, train_data: np.ndarray) -> "ESN":
        """Train ESN readout on training trajectory.

        Parameters
        ----------
        train_data : ndarray, shape (T, d)
            Training time series.
        """
        T, d = train_data.shape
        self.d_out = d

        if self.Win is None or self.Win.shape[1] != d:
            self._build_reservoir(d)

        # Run reservoir
        states = self._run_reservoir(train_data)

        # Discard washout steps
        wo = min(self.washout, T // 4)
        S = states[wo:]        # (T-wo, N)
        Y = train_data[wo:]    # (T-wo, d)  — target is the NEXT step (shift by 1)

        # Use states[wo:-1] to predict train_data[wo+1:]
        if len(S) > 1:
            S_in = S[:-1]   # (T-wo-1, N)
            Y_out = Y[1:]   # (T-wo-1, d)
        else:
            S_in = S
            Y_out = Y

        # Ridge regression: Wout = (S'S + ridge*I)^{-1} S' Y
        A = S_in.T @ S_in + self.ridge * np.eye(self.N)
        B = S_in.T @ Y_out
        self.Wout = np.linalg.solve(A, B)  # (N, d)

        # Store last reservoir state for prediction
        self._last_state = states[-1].copy()
        self._last_input = train_data[-1].copy()

        return self

    def predict(self, n_steps: int, init_data: np.ndarray | None = None) -> np.ndarray:
        """Generate autoregressive predictions.

        Parameters
        ----------
        n_steps : int
            Number of steps to predict.
        init_data : ndarray, shape (T_init, d), optional
            If given, run reservoir warm-up on this data before predicting.

        Returns
        -------
        predictions : ndarray, shape (n_steps, d)
        """
        if init_data is not None:
            states = self._run_reservoir(init_data)
            x = states[-1].copy()
            u = init_data[-1].copy()
        else:
            x = self._last_state.copy()
            u = self._last_input.copy()

        preds = np.zeros((n_steps, self.d_out))

        for t in range(n_steps):
            # Advance reservoir one step
            pre = self.W @ x + self.Win @ u
            x = (1.0 - self.alpha) * x + self.alpha * np.tanh(pre)
            # Readout
            u = x @ self.Wout
            preds[t] = u

        return preds

    def fit_predict(
        self,
        train_data: np.ndarray,
        n_steps: int,
        init_data: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convenience: fit then predict."""
        self.fit(train_data)
        return self.predict(n_steps, init_data=init_data)
