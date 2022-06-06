"""
A module defining two classes (in Jax !) which can create arbitrary-dimensional fields with given power spectra. One such function
produces *Gaussian* fields, and the other *LogNormal* fields.
In principle, these may be extended to other 1-point density distributions by subclassing :class:`PowerBox` and
over-writing the same methods as are over-written in :class:`LogNormalPowerBox`.
"""
import jax
import jax.numpy as np
from powerbox_jax import dft
from powerbox_jax.tools import _magnitude_grid
import numpyro 
import numpyro.distributions as dist

def _make_hermitian(mag, pha):
    r"""
    Take random arrays and convert them to a complex hermitian array.
    Note that this assumes that mag is distributed normally.
    Parameters
    ----------
    mag : array
        Normally-distributed magnitudes of the complex vector.
    pha : array
        Uniformly distributed phases of the complex vector
    Returns
    -------
    kspace : array
        A complex hermitian array with normally distributed amplitudes.
    """
    revidx = (slice(None, None, -1),) * len(mag.shape)
    mag = (mag + mag[revidx]) / np.sqrt(2)
    pha = (pha - pha[revidx]) / 2 + np.pi
    return mag * (np.cos(pha) + 1j * np.sin(pha))


class PowerBox(object):
    r"""
    Calculate real- and fourier-space Gaussian fields generated with a given power spectrum.
    Parameters
    ----------
    N : int
        Number of grid-points on a side for the resulting box (equivalently, number of wavenumbers to use).
    pk : callable
        A callable of a single (vector) variable `k`, which is the isotropic power spectrum. The relationship of the
        `k` of which this is a function to the real-space co-ordinates, `x`, is determined by the parameters ``a,b``.
    dim : int, default 2
        Number of dimensions of resulting box.
    boxlength : float, default 1.0
        Length of the final signal on a side. This may have arbitrary units, so long as `pk` is a function of a
        variable which has the inverse units.
    ensure_physical : bool, optional
        Interpreting the power spectrum as a spectrum of density fluctuations, the minimum physical value of the
        real-space field, :meth:`delta_x`, is -1. With ``ensure_physical`` set to ``True``, :meth:`delta_x` is
        clipped to return values >-1. If this is happening a lot, consider using :class:`LogNormalPowerBox`.
    a,b : float, optional
        These define the Fourier convention used. See :mod:`powerbox.dft` for details. The defaults define the standard
        usage in *cosmology* (for example, as defined in Cosmological Physics, Peacock, 1999, pg. 496.). Standard
        numerical usage (eg. numpy) is (a,b) = (0,2pi).
    vol_normalised_power : bool, optional
        Whether the input power spectrum, ``pk``, is volume-weighted. Default True because of standard cosmological
        usage.
    Notes
    -----
    A number of conventions need to be listed.
    The conventions of using `x` for "real-space" and `k` for "fourier space" arise from cosmology, but this does
    not affect anything -- `x` could just as well stand for "time domain" and `k` for "frequency domain".
    The important convention is the relationship between `x` and `k`, or in other words, whether `k` is interpreted
    as an angular frequency or ordinary frequency. By default, because of cosmological conventions, `k` is an
    angular frequency, so that the fourier transform integrand is delta_k*exp(-ikx). The conventions can be changed
    arbitrarily by setting the ``a,b`` parameters (see :mod:`powerbox.dft` for details).
    The primary quantity of interest is :meth:`delta_x`, which is a zero-mean Gaussian field with a power spectrum
    equivalent to that which was input. Being zero-mean enables its direct interpretation as an overdensity
    field, and this interpretation is enforced in the :meth:`make_discrete_sample` method.
    .. note:: None of the n-dimensional arrays that are created within the class are stored, due to the inefficiency
              in memory consumption that this would imply. Thus, each large array is created and *returned* by their
              respective method, to be stored/discarded by the user.
    Examples
    --------
    To create a 3-dimensional box of gaussian over-densities, gridded into 100 bins, with cosmological conventions,
    and a power-law power spectrum, simply use
    >>> pb = PowerBox(100,lambda k : 0.1*k**-3., dim=3, boxlength=100.0)
    >>> overdensities = pb.delta_x()
    >>> grid = pb.x
    >>> radii = pb.r
    To create a 2D turbulence structure, with arbitrary units, once can use
    >>> import matplotlib.pyplot as plt
    >>> pb = PowerBox(1000, lambda k : k**-7./5.)
    >>> plt.imshow(pb.delta_x())
    """

    def __init__(self, N, pk, dim=2, boxlength=1.0, supplied_freqs=None, ensure_physical=False, a=1., b=1.,
                 vol_normalised_power=True):

        self.N = N
        self.dim = dim
        self.boxlength = boxlength
        self.L = boxlength
        self.fourier_a = a
        self.fourier_b = b
        self.vol_normalised_power = vol_normalised_power
        self.V = self.boxlength ** self.dim

        if self.vol_normalised_power:
            self.pk = lambda k: pk(k) / self.V
        else:
            self.pk = pk

        self.ensure_physical = ensure_physical
        self.Ntot = self.N ** self.dim

        if N % 2 == 0:
            self._even = True
        else:
            self._even = False

        self.n = N + 1 if self._even else N

        # Get the grid-size for the final real-space box.
        self.dx = float(boxlength) / N

        # shape of powerbox
        self.shape = (N,)*self.dim

        if supplied_freqs is None:
          self.freqs,self.axes,self.left_edge = self.get_freqs()

        else:
          self.freqs,self.axes,self.left_edge = supplied_freqs


    def k(self):
        "The entire grid of wavenumber magitudes"
        return _magnitude_grid(self.kvec, self.dim)


    def get_freqs(self):
        "set frequencies for jittable fft calculation"

        axes = list(range(len(self.shape)))
        _N = np.array([self.shape[axis] for axis in axes])

        # Get the box volume if given the real-space box volume
        _L = np.array([self.L] * len(axes))
        dx = _L / _N
        Lk = 2 * np.pi / (dx * self.fourier_b)

        Lk = np.array(Lk)
        left_edge = dft._set_left_edge(None, axes, Lk)

        V = np.product(Lk)
        dk = np.array(Lk) / np.array(_N)

        _myfreq = lambda n,d: dft.fftfreq(n, d=d, b=self.fourier_b)
        freq = jax.tree_multimap(_myfreq, list(self.shape), list(dk))
        return freq, axes, left_edge

    @property
    def kvec(self):
        "The vector of wavenumbers along a side"
        return dft.fftfreq(self.N, d=self.dx, b=self.fourier_b)

    @property
    def r(self):
        "The radial position of every point in the grid"
        return _magnitude_grid(self.x, self.dim)

    @property
    def x(self):
        "The co-ordinates of the grid along a side"
        return np.arange(-self.boxlength / 2, self.boxlength / 2, self.dx)[:self.N]

    def gauss_hermitian(self):
        "A random array which has Gaussian magnitudes and Hermitian symmetry"
        shape = (self.n,) * self.dim
        mag = numpyro.sample('gauss_hermitian_mag', dist.Independent(dist.Normal(np.zeros(shape), 
                                                                                np.ones(shape)) ,
                                                                     self.dim))
        pha = numpyro.sample('gauss_hermitian_pha', dist.Independent(dist.Uniform(np.zeros(shape), 
                                                                 2 * np.pi * np.ones(shape)),
                                                                     self.dim))

        dk = _make_hermitian(mag, pha)

        if self._even:
            cutidx = (slice(None, -1),) * self.dim
            dk = dk[cutidx]

        return dk

    def power_array(self):
        "The Power Spectrum (volume normalised) at `self.k`"
        k = self.k()
        mask = (self.n // 2,)*self.dim #np.where(k == 0)

        # replace monopole mode with one for stability
        k = k.at[mask].set(np.array(1.))
        k = self.pk(k)

        # replace monopole mode with zero
        k = k.at[mask].set(np.array(0.))
        return k

    def delta_k(self):
        "A realisation of the delta_k, i.e. the gaussianised square root of the power spectrum (i.e. the Fourier co-efficients)"
        p = self.power_array()

        #if np.any(p < 0):
        #    raise ValueError("The power spectrum function has returned negative values.")

        # here we mask out the monopole so that the derivatives of p(k)
        # stay stable when we set p(k=0)=0.

        mask = (self.n // 2,)*self.dim
        p = p.at[mask].set(np.array(1.))
        p = np.sqrt(p)
        p = p.at[mask].set(np.array(0.))

        gh = self.gauss_hermitian()
        gh *= p
        return gh

    def delta_x(self):
        "The realised field in real-space from the input power spectrum"
        # Here we multiply by V because the (inverse) fourier-transform of the (dimensionless) power has
        # units of 1/V and we require a unitless quantity for delta_x.
        dk = self.delta_k()
        dk = self.V * dft.ifft(dk, L=self.boxlength, freq=self.freqs, left_edge=self.left_edge,
                           a=self.fourier_a, b=self.fourier_b)[0]
        dk = np.real(dk)

        if self.ensure_physical:
            np.clip(dk, -1, np.inf, dk)

        return dk

    def create_discrete_sample(self, nbar, randomise_in_cell=True, min_at_zero=False,
                               store_pos=False):
        r"""
        Assuming that the real-space signal represents an over-density with respect to some mean, create a sample
        of tracers of the underlying density distribution.
        Parameters
        ----------
        nbar : float
            Mean tracer density within the box.
        randomise_in_cell : bool, optional
            Whether to randomise the positions of the tracers within the cells, or put them at the grid-points (more
            efficient).
        min_at_zero : bool, optional
            Whether to make the lower corner of the box at the origin, otherwise the centre of the box is at the
            origin.
        store_pos : bool, optional
            Whether to store the sample of tracers as an instance variable `tracer_positions`.
        Returns
        -------
        tracer_positions : float, array_like
            ``(n, d)``-array, with ``n`` the number of tracers and ``d`` the number of dimensions. Each row represents
            a single tracer's co-ordinates.
        """
        dx = self.delta_x()
        dx = (dx + 1) * self.dx ** self.dim * nbar
        n = dx
        self.n_per_cell = numpyro.sample('n_per_cell', dist.Poisson(n))

        # Get all source positions
        args = [self.x] * self.dim
        X = np.meshgrid(*args)

        tracer_positions = np.array([x.flatten() for x in X]).T
        tracer_positions = tracer_positions.repeat(self.n_per_cell.flatten(), axis=0)

        if randomise_in_cell:
            ntot = np.sum(self.n_per_cell)
            tracer_positions += numpyro.sample('tracer_shifts', dist.Uniform(np.zeros(ntot, self.dim), np.ones(ntot, self.dim)* self.dx ))

        if min_at_zero:
            tracer_positions += self.boxlength / 2.0

        if store_pos:
            self.tracer_positions = tracer_positions

        return tracer_positions


class LogNormalPowerBox(PowerBox):
    r"""
    Calculate Log-Normal density fields with given power spectra.
    See the documentation of :class:`PowerBox` for a detailed explanation of the arguments, as this class
    has exactly the same arguments.
    This class calculates an (over-)density field of arbitrary dimension given an input isotropic power spectrum. In
    this case, the field has a log-normal distribution of over-densities, always yielding a physically valid field.
    Examples
    --------
    To create a log-normal over-density field:
    >>> from powerbox import LogNormalPowerBox
    >>> lnpb = LogNormalPowerBox(100,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> overdensities = lnpb.delta_x
    >>> grid = lnpb.x
    >>> radii = lnpb.r
    To plot the overdensities:
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(pb.delta_x)
    Compare the fields from a Gaussian and Lognormal realisation with the same power:
    >>> lnpb = LogNormalPowerBox(300,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> pb = PowerBox(300,lambda k : k**-7./5.,dim=2, boxlength=1.0)
    >>> fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,5))
    >>> ax[0].imshow(lnpb.delta_x,aspect="equal",vmin=-1,vmax=lnpb.delta_x.max())
    >>> ax[1].imshow(pb.delta_x,aspect="equal",vmin=-1,vmax = lnpb.delta_x.max())
    To create and plot a discrete version of the field:
    >>> positions = lnpb.create_discrete_sample(nbar=1000.0, # Number density in terms of boxlength units
    >>>                                         randomise_in_cell=True)
    >>> plt.scatter(positions[:,0],positions[:,1],s=2,alpha=0.5,lw=0)
    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    def correlation_array(self):
        "The correlation function from the input power, on the grid"
        pa = self.power_array()
        return self.V * np.real(dft.ifft(pa, L=self.boxlength, freq=self.freqs, left_edge=self.left_edge,
                                              a=self.fourier_a, b=self.fourier_b)[0])

    def gaussian_correlation_array(self):
        "The correlation function required for a Gaussian field to produce the input power on a lognormal field"
        return np.log(1 + self.correlation_array())

    def gaussian_power_array(self):
        "The power spectrum required for a Gaussian field to produce the input power on a lognormal field"
        gca = self.gaussian_correlation_array()
        gpa = np.abs(dft.fft(gca, L=self.boxlength, freq=self.freqs, left_edge=self.left_edge,
                                                  a=self.fourier_a, b=self.fourier_b)[0])

        mask = (self.n // 2,)*self.dim
        gpa = gpa.at[mask].set(np.array(1e-12))

        return gpa

    def delta_k(self):
        """
        A realisation of the delta_k, i.e. the gaussianised square root of the unitless power spectrum
        (i.e. the Fourier co-efficients)
        """
        p = self.gaussian_power_array()
        gh = self.gauss_hermitian()
        gh = np.sqrt(p) * gh
        return gh

    def delta_x(self):
        "The real-space over-density field, from the input power spectrum"
        dk = self.delta_k()
        dk = np.sqrt(self.V) * dft.ifft(dk, L=self.boxlength, freq=self.freqs, left_edge=self.left_edge,
                                    a=self.fourier_a, b=self.fourier_b)[0]
        dk = np.real(dk)

        sg = np.var(dk)
        return np.exp(dk - sg / 2) - 1.
