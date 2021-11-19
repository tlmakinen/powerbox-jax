# powerbox-jax
`powerbox` + `Jax` ... like PB&J, an ideal combo

Jax implementation of `powerbox` (https://github.com/steven-murray/powerbox) for autodifferentiability.
 `powerbox-jax` is functionally equivalent to `powerbox`, but is now fully differentiable and XLA-compatible !

## installation

For installation on the command line or Colab, call

`pip install git+https://github.com/tlmakinen/powerbox-jax.git`


# example: generate a differentiable mock dark matter field
In computational cosmology it is of interest to take gradients of cosmological *fields* with respect to underlying global parameters. Doing so lets us efficiently calculate information content and train neural networks via gradient descent (see e.g.https://arxiv.org/abs/2107.07405 ).

For this example we'll need to install the `imnn` and `jax-cosmo` packages.

`pip install imnn`
and
`pip install jax-cosmo`

Now onto the code:

