"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import einops
from torch.distributions import MultivariateNormal

from optimal_weights import omega_optimized, gamma_by_gamma_max, gamma_by_r


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N, K=0, H=0.5, gamma_max=20.0, norm_var=True, device='cpu'):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
          K: number of augmenting OU processes; K=0 corresponds to the purely Brownian case
          H: Hurst index of MA-fBM
          gamma_max: maximal space grid point; corresponds the maximal speed of mean reversion for the augmenting OU processes
          norm_var: normalize the terminal and initial marginal variance to the corresponding values of the purely Brownian case
        """

        super().__init__()
        self.N = N
        self.K = K
        self.H = H
        self.gamma_max = gamma_max
        self.norm_var = norm_var
        self.device = device

        '''
        Given H and K determine the optimal approximation coefficients by solving Aw=b for w
        '''

        if self.K > 0:
            if self.K == 1:
                gamma = gamma_by_r(K, torch.sqrt(torch.tensor(gamma_max)), device=device)
            else:
                gamma = gamma_by_gamma_max(K, self.gamma_max, device=device)
            omega, A, b = omega_optimized(
                gamma, self.H, self.T, return_Ab=True, device=device
            )

        else:
            gamma = torch.tensor([0.0])
            omega = torch.tensor([1.0])
            A = torch.tensor([1.0])
            b = torch.tensor([1.0])

        self.register_buffer("gamma", torch.as_tensor(gamma, device=device)[None, :])
        self.register_buffer("gamma_i", self.gamma[:, :, None].clone())
        self.register_buffer("gamma_j", self.gamma[:, None, :].clone())
        self.update_omega(omega,A=A,b=b)

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    @abstractmethod
    def compute_cov(self,t):
        '''
        Compute covariance matrix of augmented forward process
        '''
        pass

    def update_omega(self,omega,A=None,b=None):

        if A is not None:
            self.register_buffer("A", torch.as_tensor(A, device=self.device))
        if b is not None:
            self.register_buffer("b", torch.as_tensor(b, device=self.device))

        self.register_buffer("omega", torch.as_tensor(omega, device=self.device)[None, :].clone())
        self.register_buffer('sum_omega', torch.sum(self.omega))
        self.register_buffer("omega_i", self.omega[:, :, None].clone())
        self.register_buffer("omega_j", self.omega[:, None, :].clone())
        self.double_sum_omega = torch.sum(self.omega_i * self.omega_j, dim=(1, 2))

    def mean_scale(self, t):
        return torch.exp(self.integral(t))

    def mean(self,x0,t):
        c_t = self.mean_scale(t)[:,None,None,None,None]
        bs,c,h,w = x0.shape
        return torch.cat([(c_t*x0[:,:,:,:,None]),torch.zeros(bs,c,h,w,self.K,device=x0.device)],dim=-1)

    def marginal_stats(self,t, batch=None, eps_pd=1e-4):

        mean = self.mean(batch, t) if batch is not None else None
        cov = self.cov(t)
        bs = cov.shape[0]
        sigma_t = torch.squeeze(cov).clone().to(t.device)

        if bs==1:
            sigma_t = sigma_t[None, :, :]

        I_eps = torch.eye(self.aug_dim, self.aug_dim,device=t.device)[None, :, :] * torch.ones((t.shape[0],self.aug_dim, self.aug_dim),device=t.device)
        I_eps[:,1:,1:] = I_eps[:,1:,1:] * (eps_pd * torch.exp(-2 * self.gamma * t[:,None])[:,:,None])
        I_eps[:, 0, 0] = 0.0
        sigma_t = sigma_t + I_eps

        corr = sigma_t[:,1:,0].clone()
        cov_yy = sigma_t[:,1:,1:].clone()
        var_x = sigma_t[:,0,0].clone()
        eta = torch.linalg.solve(cov_yy,corr)
        var_c = torch.sum(eta*corr,dim=-1)
        return sigma_t[:,None,None,None], mean, corr, cov_yy, eta[:,None,None,None,:], var_x[:,None,None,None], var_c[:,None,None,None]

    def compute_YiYj(self,t):
        sum_gamma = self.gamma_i + self.gamma_j
        return ((1-torch.exp(-t*sum_gamma))/sum_gamma)

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * \
                    score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * \
                    score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

'''
#Purely Brownian original code
class RVESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, T=1):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N
        self.T_val = T

    @property
    def T(self):
        return self.T_val

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.rand(*shape)

    def prior_logp(self, z):
        return torch.zeros_like(z)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas.to(t.device)[timestep - 1])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
 '''

class RVESDE(SDE):
    def __init__(
            self,
            sigma_min=0.01,
            sigma_max=50,
            N=1000,
            T=1,
            K=0,
            H=0.5,
            gamma_max=20.0,
            norm_var=True,
            device="cpu",
    ):
        """Construct a Fractional Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
          K: number of augmenting OU processes; K=0 corresponds to the purely Brownian case
          H: Hurst index of MA-fBM
          gamma_max: maximal space grid point; corresponds the maximal speed of mean reversion for the augmenting OU processes
          norm_var: normalize the terminal and initial marginal variance to the corresponding values of the purely Brownian case
        """

        super().__init__(
            N, K=K, H=H, gamma_max=gamma_max, norm_var=norm_var, device=device
        )
        self.register_buffer("sigma_min", torch.as_tensor(torch.tensor([sigma_min]), device=self.device))
        self.register_buffer("sigma_max", torch.as_tensor(torch.tensor([sigma_max]), device=self.device))

        self.register_buffer("r", torch.as_tensor(self.sigma_max / self.sigma_min, device=self.device))
        self.register_buffer("a", torch.as_tensor(
            self.sigma_min * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min))),
            device=self.device))

        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )

        self.N = N
        self.T_val = T

        '''normalize the terminal and initial marginal variance to the corresponding values of the purely Brownian case'''

        if self.norm_var and self.K > 0:
            var_T = self.compute_covXiXj(self.T[:, None, None])
            omega = self.sigma_max * self.omega[0] / torch.sqrt(var_T)
            self.update_omega(omega)

    @property
    def T(self):
        return self.T_val

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.rand(*shape)

    def prior_logp(self, z):
        return torch.zeros_like(z)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas.to(t.device)[timestep - 1])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G

    def mu(self, t):
        return 0 * t

    def g(self, t):
        return self.a * (self.r ** t)

    def integral(self, t):
        return 0 * t

    def brown_var(self, t):
        return (self.sigma_min ** 2) * ((self.sigma_max / self.sigma_min) ** (2 * t))  # - self.sigma_min**2

    def terminal(self, z):

        shape = z.shape
        if len(shape) == 4:
            log_prob = self.prior_logp(z)
        else:
            bs, c, h, w, aug_dim = z.shape
            sigma_T = torch.squeeze(self.marginal_stats(self.T)[0])
            mvn = MultivariateNormal(torch.zeros(aug_dim, device=z.device), sigma_T)
            z = einops.rearrange(z, 'B C H W K->(B C H W) K ', B=bs, C=c, H=h, W=w, K=aug_dim)
            log_prob = mvn.log_prob(z)
            log_prob = einops.rearrange(log_prob, '(B C H W) -> B (C H W)', B=bs, C=c, H=h, W=w)
            log_prob = torch.sum(log_prob, dim=1)
        return log_prob

    def compute_cov(self, t):
        bs = t.shape[0]
        sigma_t = torch.zeros(bs, self.aug_dim, self.aug_dim)
        XYk = self.compute_XYl(t, self.omega_i, self.gamma_i, self.gamma_j)
        sigma_t[:, 0, 0] = self.compute_covXX(t)
        sigma_t[:, 1:, 0] = XYk.clone()
        sigma_t[:, 0, 1:] = XYk.clone()
        sigma_t[:, 1:, 1:] = self.compute_YiYj(t)

        return sigma_t[:, None, None, None, :, :]

    def cov(self, t):
        t = t[None, None, None] if len(t.shape) == 0 else t[:, None, None]
        sigma_t = self.compute_cov(t)
        return sigma_t

    def compute_Ik(self, t, gamma_k):

        a = self.a.clone()
        r = self.r.clone()

        part1 = (((a ** 2) * gamma_k) / (torch.log(r) - gamma_k)) * (
                ((r ** (2 * t)) - ((r ** t) * torch.exp(-gamma_k * t)))
                / (torch.log(r) + gamma_k)
        )
        part2 = (((a ** 2) * gamma_k) / (torch.log(r) - gamma_k)) * (
                (r ** (2 * t) - 1) / (2 * torch.log(r))
        )
        return part1 - part2

    def compute_Iij(self, t, gamma_i, gamma_j):

        a = self.a.clone()
        r = self.r.clone()

        scale = ((a ** 2) * (gamma_i * gamma_j)) / (
                (torch.log(r) - gamma_i) * (torch.log(r) - gamma_j)
        )
        part1 = (r ** (2 * t)) * (
                (1 - torch.exp(-(gamma_i + gamma_j) * t)) / (gamma_i + gamma_j)
        )
        part2 = ((r ** (2 * t)) - ((r ** t) * torch.exp(-gamma_i * t))) / (
                torch.log(r) + gamma_i
        )
        part3 = ((r ** (2 * t)) - ((r ** t) * torch.exp(-gamma_j * t))) / (
                torch.log(r) + gamma_j
        )
        part4 = ((r ** (2 * t)) - 1) / (2 * torch.log(r))
        return scale * (part1 - part2 - part3 + part4)

    def compute_covXiXj(self, t):

        '''implmentation of eq. (195-201) using eq. (202) to calculate self.brown_var'''
        offset = self.sigma_min ** 2
        Ii = self.compute_Ik(t[:, :, 0], self.gamma_i[:, :, 0])[:, :, None]
        Ij = self.compute_Ik(t[:, 0, :], self.gamma_j[:, 0, :])[:, None, :]
        Iij = self.compute_Iij(t, self.gamma_i, self.gamma_j)
        omega_ij = self.omega_i * self.omega_j
        return torch.sum(omega_ij * (self.brown_var(t) - offset + (Iij - (Ii + Ij))), dim=(1, 2)) + offset

    def compute_covXX(self, t):
        return self.compute_covXiXj(t)

    def compute_XYl(self, t, omega_k, gamma_k, gamma_l):

        a = self.a.clone()
        r = self.r.clone()

        part1 = (a / (torch.log(r) + gamma_l)) * ((r ** t) - torch.exp(-gamma_l * t))
        part2 = (
                ((a * gamma_k) / (torch.log(r) - gamma_k))
                * ((r ** t) / (gamma_k + gamma_l))
                * (1 - torch.exp(-(gamma_k + gamma_l) * t))
        )
        part3 = (
                ((a * gamma_k) / (torch.log(r) - gamma_k))
                * (1 / (torch.log(r) + gamma_l))
                * ((r ** t) - torch.exp(-gamma_l * t))
        )
        return torch.sum(omega_k * (part1 - part2 + part3), dim=1)