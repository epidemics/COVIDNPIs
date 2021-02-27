import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# def HalfStudentT(nu, sigma):
#    return dist.continuous.TruncatedDistribution(
#        dist.StudentT(df=nu, scale=sigma), lo=0.0
#    )


def sample_half_student_t(name, nu, sigma):
    t = numpyro.sample(f"{name}/t", dist.StudentT(df=nu, scale=sigma))
    return numpyro.deterministic(name, jnp.maximum(t, -t))


def sample_asymmetric_laplace(name, scale, kappa):
    """
    Asymmetric laplace distribution as in https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution with m = 0.0
    """
    return numpyro.deterministic(
        name,
        numpyro.sample(f"{name}/a", dist.Exponential(0.0, scale / -kappa))
        - numpyro.sample(f"{name}/b", dist.Exponential(0.0, scale * kappa)),
    )
