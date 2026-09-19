"""SKIRT -> ground-based r-band observation forward model.

The SKIRT cube is a one-second exposure in electron counts, so its r-band
plane is an electron-rate image (e/s) at ``z_source``.  Sources are moved to a
single fixed target redshift by the luminosity-distance ratio only: no K
correction, no pixel-scale change.  Everything downstream is a deliberately
imperfect ground pipeline -- real sky level, spatially varying sky, PSF
convolution, Poisson and read noise, and a sky that is *estimated from the
data* rather than subtracted exactly.

Three planes are delivered per observation, all in nanomaggies
(m_AB = 22.5 - 2.5 log10 f), converted from e/s at the configured zeropoint:

``OBS``    background-subtracted observation, as a real pipeline would deliver
``GT``     noiseless PSF-convolved truth (data construction + final test only)
``SIGMA``  per-pixel 1-sigma uncertainty

SIGMA carries the *total* error budget with no separation of source and
background terms and no mask: source shot noise, sky shot noise, read noise,
and the uncertainty incurred by estimating and subtracting the background are
propagated into one variance and delivered as a single plane.  Its Poisson
terms are estimated only from pipeline-available measurements (the noisy
science exposure and fitted background), never from GT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

# One worker == one core.  Stop NumPy/GalSim from multiplying that internally.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import galsim
import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

R_BAND_INDEX = 3  # FILTER = 'NUV,u,g,r,i,z,y'
SKIRT_PIXEL_SCALE = 0.4625279505845532  # arcsec/px, header PS at z=0.01


@dataclass(frozen=True)
class Params:
    """Physical and instrumental knobs exposed to the tuning notebook."""

    # --- cosmology / redshift -------------------------------------------
    z_source: float = 0.01          # SKIRT native redshift (from header)
    z_target: float = 0.10          # single fixed target redshift
    h0: float = 70.0
    omega_m: float = 0.30
    omega_lambda: float = 0.70

    # --- exposure --------------------------------------------------------
    n_exposures: int = 3
    exposure_time: float = 60.0     # seconds per exposure

    # --- sky -------------------------------------------------------------
    sky_mag: float = 21.0           # r mag / arcsec^2, dark-ish ground site
    zeropoint: float = 25.0         # mag -> e/s at the delivered pixel scale
    # Sky is a plane tilted in a random direction across the field.  Flat-field
    # and illumination residuals on a real r-band frame are a few tenths of a
    # percent of sky, not a few percent: sky is ~8.5 e/s here, so 2% of it
    # would be 0.17 e/s -- comparable to the 0.23 e/s pixel noise, which would
    # cover the frame in visible blotches.
    sky_tilt_amp: float = 0.002       # linear term, fraction of sky

    # --- PSF -------------------------------------------------------------
    seeing_fwhm: float = 1.10       # arcsec
    moffat_beta: float = 3.5

    # --- detector --------------------------------------------------------
    read_noise: float = 7.0         # electrons per exposure per pixel
    gain: float = 1.0               # e/ADU; counts are already electrons

    # --- background estimation (deliberately imperfect) ------------------
    bkg_box_size: int = 64          # pipeline mesh; larger => more residual
    bkg_filter_size: int = 3
    bkg_sigma_clip: float = 3.0

    # --- geometry --------------------------------------------------------
    pixel_scale: float = 0.4625279505845532  # arcsec/px of the delivered image

    # Angular size handling.  The SKIRT cube is sampled at 0.4625"/px because
    # that is what 100 pc subtends at z_source=0.01.  Moving the source to
    # z_target should shrink it by D_A(z_src)/D_A(z_tgt) ~ 9x at z=0.1.
    #   False -> keep the native angular size ("do not consider pixel scale",
    #            the literal reading of the spec; galaxies stay large and end
    #            up at very low surface brightness)
    #   True  -> apply the angular-diameter shrink, which is what a real
    #            z_target galaxy would subtend
    apply_angular_scaling: bool = False


def luminosity_distance_ratio(p: Params) -> float:
    """Flux scaling from ``z_source`` to ``z_target``: (D_src / D_tgt)^2."""
    cosmo = LambdaCDM(H0=p.h0, Om0=p.omega_m, Ode0=p.omega_lambda)
    d_src = cosmo.luminosity_distance(p.z_source).value
    d_tgt = cosmo.luminosity_distance(p.z_target).value
    return float((d_src / d_tgt) ** 2)


def angular_shrink(p: Params) -> float:
    """Linear angular-size ratio D_A(z_src)/D_A(z_tgt); >1 means it shrinks.

    Returns 1.0 when angular scaling is switched off, in which case the source
    keeps its native z_source angular extent.
    """
    if not p.apply_angular_scaling:
        return 1.0
    cosmo = LambdaCDM(H0=p.h0, Om0=p.omega_m, Ode0=p.omega_lambda)
    da_src = cosmo.angular_diameter_distance(p.z_source).value
    da_tgt = cosmo.angular_diameter_distance(p.z_target).value
    return float(da_tgt / da_src)


def load_r_band(path) -> np.ndarray:
    """Read the r-band plane (e/s for a 1 s exposure) from a SKIRT cube."""
    with fits.open(path, memmap=False) as hdul:
        cube = np.asarray(hdul[1].data, dtype=np.float64)
    return np.ascontiguousarray(cube[R_BAND_INDEX])


def sky_rate(p: Params) -> float:
    """Sky brightness -> e/s per pixel."""
    area = p.pixel_scale ** 2  # arcsec^2 per pixel
    return float(10.0 ** (-0.4 * (p.sky_mag - p.zeropoint)) * area)


def nanomaggy_per_rate(p: Params) -> float:
    """e/s -> nanomaggy conversion for the configured zeropoint.

    A rate R e/s has magnitude m = zeropoint - 2.5 log10(R); nanomaggies are
    defined by m = 22.5 - 2.5 log10(f).  Hence f = R * 10^(0.4*(22.5 - zp)).
    """
    return float(10.0 ** (0.4 * (22.5 - p.zeropoint)))


def _sky_plane(shape, p: Params, rng: np.random.Generator) -> np.ndarray:
    """Spatially varying sky as a plane tilted in a random direction, mean 1.0.

    A first-order gradient is what a real flat-fielded frame's residual
    illumination looks like at this scale; anything finer is absorbed by the
    pipeline's background mesh anyway.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    # Normalised to [-1, 1] across the field so the amplitude is the
    # peak-to-centre fractional swing.
    yy = 2.0 * (yy - 0.5 * ny) / max(ny, 1)
    xx = 2.0 * (xx - 0.5 * nx) / max(nx, 1)

    angle = rng.uniform(0.0, 2.0 * np.pi)
    tilt = np.cos(angle) * xx + np.sin(angle) * yy

    field = 1.0 + p.sky_tilt_amp * tilt
    return field / float(field.mean())


def _psf(p: Params) -> galsim.GSObject:
    return galsim.Moffat(beta=p.moffat_beta, fwhm=p.seeing_fwhm)


def estimate_exposure_variance(
    residual_e: np.ndarray,
    background_e: np.ndarray,
    read_noise: float,
    background_var_e: np.ndarray,
) -> np.ndarray:
    """Estimate a science exposure's variance without access to GT.

    A calibrated CCD pipeline has the background-subtracted science image,
    its fitted background model, and detector read noise.  It estimates the
    non-negative source counts as ``max(science - background, 0)`` and uses

        var = source_est + background_est + read_noise^2 + var(background_est)

    in electron units.  Clipping the source estimate at zero prevents a
    downward noise fluctuation from claiming sub-sky Poisson variance.
    """
    source_est_e = np.clip(np.asarray(residual_e, dtype=np.float64), 0.0, None)
    sky_est_e = np.clip(np.asarray(background_e, dtype=np.float64), 0.0, None)
    bkg_var_e = np.clip(np.asarray(background_var_e, dtype=np.float64), 0.0, None)
    return source_est_e + sky_est_e + float(read_noise) ** 2 + bkg_var_e


def render_truth(rate_native: np.ndarray, p: Params, coverage: np.ndarray | None = None):
    """PSF-convolved, redshift-scaled, noiseless source rate image (e/s).

    This is the GT plane.  It never enters the observation path.

    If ``coverage`` is given (same shape as ``rate_native``, 1.0 where a pixel
    is real native SKIRT data and 0.0 where it is synthetic zero-padding), a
    second array is returned alongside GT: that indicator run through the
    *exact same* geometric mapping (angular scaling, PSF convolution, canvas
    sampling) as the flux itself.  This is the only correct way to know which
    final-canvas pixels are contaminated by padding -- the PSF convolution
    smears the pad boundary, so a naive re-derivation of the boundary in
    output-pixel coordinates would not match where flux actually leaked in
    from the padded zeros.  The returned coverage is fractional in (0, 1) at
    the boundary, not a binary mask: a pixel whose PSF footprint is half in
    real data and half in padding is genuinely half-trustworthy.
    """
    scaled = rate_native * luminosity_distance_ratio(p)
    ny, nx = scaled.shape

    # The cube's own sampling is whatever 100 pc subtends at z_source.  Placing
    # the source at z_target divides its angular extent by ``shrink``; total
    # flux is unchanged by this (it is set by the D_L ratio above), so the
    # InterpolatedImage keeps flux normalization while the scale shrinks.
    shrink = angular_shrink(p)
    src_scale = SKIRT_PIXEL_SCALE / shrink

    src = galsim.InterpolatedImage(
        galsim.Image(np.ascontiguousarray(scaled), scale=src_scale),
        normalization="flux",
        x_interpolant="lanczos3",
        calculate_stepk=False,
        calculate_maxk=False,
    )
    convolved = galsim.Convolve([src, _psf(p)])

    # Canvas covers exactly the SKIRT field of view on the delivered pixel
    # grid -- no padding.  Padding would surround the cube's faint outskirts
    # (still ~5e-4 e/s at the edge) with exact zeros, and the arcsinh stretch
    # turns that three-order-of-magnitude step into a hard square border in GT.
    side = int(np.ceil(max(ny, nx) * src_scale / p.pixel_scale))
    canvas = galsim.Image(side, side, scale=p.pixel_scale)
    convolved.drawImage(image=canvas, method="no_pixel")
    gt = np.asarray(canvas.array, dtype=np.float64)

    if coverage is None:
        return gt

    cov_src = galsim.InterpolatedImage(
        galsim.Image(np.ascontiguousarray(coverage.astype(np.float64)), scale=src_scale),
        normalization="flux",
        x_interpolant="lanczos3",
        calculate_stepk=False,
        calculate_maxk=False,
    )
    # Convolve with the same PSF so a boundary pixel's coverage reflects how
    # much of its own PSF footprint actually came from real data, then
    # renormalise: InterpolatedImage(normalization="flux") preserves the
    # input's *sum*, not its [0, 1] range, so drawing it like a flux map would
    # rescale coverage by the source's pixel area instead of leaving it as a
    # fraction.
    cov_canvas = galsim.Image(side, side, scale=p.pixel_scale)
    galsim.Convolve([cov_src, _psf(p)]).drawImage(image=cov_canvas, method="no_pixel")
    coverage_out = np.asarray(cov_canvas.array, dtype=np.float64) * (src_scale / p.pixel_scale) ** 2
    return gt, np.clip(coverage_out, 0.0, 1.0)


def bin_planes(obs: np.ndarray, gt: np.ndarray, sigma: np.ndarray, factor: int = 2,
               coverage: np.ndarray | None = None):
    """Flux-conserving ``factor x factor`` binning of the delivered planes.

    OBS and GT are per-pixel rates (e/s); a binned superpixel carries the sum
    of its children, so total flux is conserved and per-pixel SNR rises by
    ``factor``.  SIGMA sums in quadrature, which is exact for the independent
    shot and read noise and mildly conservative for the smooth background-
    estimate term.  Trailing rows/columns that do not fill a superpixel are
    dropped.

    ``coverage``, if given, is *averaged* (not summed) over each superpixel,
    since it is already a fraction in [0, 1] rather than an extensive flux
    quantity -- a binned pixel's trustworthiness is the mean trustworthiness
    of its children.
    """
    f = int(factor)

    def _sum(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
        return a[:ny, :nx].reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))

    out = _sum(obs), _sum(gt), np.sqrt(_sum(np.asarray(sigma, dtype=np.float64) ** 2))
    if coverage is None:
        return out
    return (*out, _sum(coverage) / float(f * f))


def simulate(rate_native: np.ndarray, p: Params, rng: np.random.Generator,
             coverage: np.ndarray | None = None):
    """Run the full forward model.  Returns (obs, gt, sigma, info) in nanomaggies.

    If ``coverage`` is given (1.0 = real native pixel, 0.0 = zero-padded), a
    fifth array is returned: that same indicator mapped onto the output
    canvas, for excluding padded pixels from test-time evaluation.  It is
    never used inside the observation path itself -- padding is invisible to
    the physics, it is only tracked for bookkeeping.
    """
    if coverage is None:
        truth_rate = render_truth(rate_native, p)          # e/s, noiseless
        coverage_out = None
    else:
        truth_rate, coverage_out = render_truth(rate_native, p, coverage=coverage)
    shape = truth_rate.shape
    t = float(p.exposure_time)
    n = int(p.n_exposures)

    sky_r = sky_rate(p)
    sky_shape = _sky_plane(shape, p, rng)              # mean ~1, varies over field
    sky_rate_map = sky_r * sky_shape                   # e/s per pixel

    # --- per-exposure detection, sky estimation and subtraction ----------
    residual_stack = []      # sky-subtracted electron counts, per exposure
    variance_stack = []      # observation-derived variance, per exposure
    for _ in range(n):
        source_e = truth_rate * t
        sky_e = sky_rate_map * t

        # Poisson on source+sky together (they are not separable on a detector),
        # then read noise.  Both land in the same electron image.
        total_e = rng.poisson(np.clip(source_e + sky_e, 0.0, None)).astype(np.float64)
        total_e += rng.normal(0.0, p.read_noise, size=shape)

        # The pipeline does not know the true sky.  It fits it on a
        # *source-free* stretch of the same detector -- the SKIRT galaxy can
        # cover the whole delivered canvas, so fitting on the science pixels
        # would swallow the source into the background model.  The companion
        # region shares the true sky surface but has its own noise; whatever
        # the fit gets wrong (estimator noise, mesh smoothing of the tilt)
        # stays in OBS.
        sky_only_e = rng.poisson(np.clip(sky_e, 0.0, None)).astype(np.float64)
        sky_only_e += rng.normal(0.0, p.read_noise, size=shape)
        bkg = Background2D(
            sky_only_e,
            box_size=(p.bkg_box_size, p.bkg_box_size),
            filter_size=(p.bkg_filter_size, p.bkg_filter_size),
            sigma_clip=SigmaClip(sigma=p.bkg_sigma_clip, maxiters=5),
            bkg_estimator=MedianBackground(),
        )
        residual_e = total_e - bkg.background
        residual_stack.append(residual_e)

        # Uncertainty of the background estimate itself.  A median over a mesh
        # cell of n_pix independent pixels carries var ~ (pi/2) * sigma^2 / n_pix,
        # and the interpolation back to full resolution keeps that scale.
        n_pix = float(p.bkg_box_size ** 2)
        cell_var = (np.pi / 2.0) * np.asarray(bkg.background_rms, dtype=np.float64) ** 2 / n_pix
        variance_stack.append(
            estimate_exposure_variance(
                residual_e=residual_e,
                background_e=bkg.background,
                read_noise=p.read_noise,
                background_var_e=cell_var,
            )
        )

    residual = np.stack(residual_stack, axis=0)
    coadd_e = residual.mean(axis=0)                    # electrons, sky-subtracted

    # --- total error budget, no source/background separation, no mask ----
    # The mean coadd is (1/n) sum_i residual_i, hence its estimated variance is
    # (1/n^2) sum_i var_i.  Every var_i above uses only the noisy science
    # exposure, the fitted background and detector metadata -- never GT.
    var_coadd_e = np.sum(np.stack(variance_stack, axis=0), axis=0) / float(n * n)

    # Convert electrons -> e/s -> nanomaggy (the delivered unit) for all
    # three planes.  The detector physics above stays in electrons; only the
    # delivery unit changes, linearly, so SIGMA converts with the same factor.
    k = nanomaggy_per_rate(p)
    obs = coadd_e / t * k
    gt = truth_rate * k
    sigma = np.sqrt(var_coadd_e) / t * k

    info = {
        "sky_rate": sky_r,
        "flux_ratio": luminosity_distance_ratio(p),
        "total_seconds": n * t,
        "sky_rate_map_mean": float(sky_rate_map.mean()),
        "nanomaggy_per_es": k,
        "sigma_method": "observed_source_plus_fitted_background",
    }
    if coverage_out is None:
        return obs, gt, sigma, info
    return obs, gt, sigma, info, coverage_out
