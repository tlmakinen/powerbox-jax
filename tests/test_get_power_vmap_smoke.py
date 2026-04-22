import jax
import jax.numpy as jnp

from powerbox_jax.tools import get_power


def _batched_get_power(x, bins=8):
    def single(field):
        n = jnp.array(field.shape)
        return get_power(
            field,
            boxlength=1.0,
            dim=2,
            bins=bins,
            N=n,
            Ndiscrete=None,
            res_ndim=2,
        )

    return jax.vmap(single)(x)


def test_get_power_vmap_smoke():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (3, 16, 16))

    pk, k = _batched_get_power(x, bins=8)

    assert pk.shape == (3, 8)
    assert k.shape == (3, 8)


def test_get_power_jit_vmap_smoke():
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (2, 16, 16))

    f = jax.jit(lambda arr: _batched_get_power(arr, bins=8))
    pk, k = f(x)

    assert pk.shape == (2, 8)
    assert k.shape == (2, 8)
