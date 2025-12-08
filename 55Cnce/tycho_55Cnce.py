import jax
import numpyro
import numpy as np
import matplotlib.pyplot as plt
import numpyro_ext
#import numpyro.distributions as dist
#import numpyro_ext.distributions as distx, numpyro_ext.optim as optimx
import arviz as az
import corner
#import itertools
import scipy
import jax.numpy as jnp
import pandas as pd 
from jaxoplanet.orbits.keplerian import Central
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.orbit import SurfaceSystem
import astropy.constants as const

from collections.abc import Callable
from functools import partial
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.orbits.keplerian import Central, Body
from jaxoplanet.types import Array, Scalar
from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.starry.core.basis import A1, A2_inv, U
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.core.rotation import left_project
from jaxoplanet.starry.core.solution import rT, solution_vector
from jaxoplanet.starry.surface import Surface 

# For multi-core parallelism (useful when running multiple MCMC chains in parallel)
numpyro.set_host_device_count(2)

# For CPU (use "gpu" for GPU)
numpyro.set_platform("cpu")

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)

h = 6.62607015e-34  # Planck constant [J*s]
c = 2.99792458e8    # speed of light [m/s]
kB = 1.380649e-23   # Boltzmann constant [J/K]

def planck(lamb_m, T):
    """Planck function B_lambda in SI units."""
    a = 2*h*c**2 / (lamb_m**5)
    b = h*c / (lamb_m*kB*T)
    return a / (jnp.exp(b) - 1)

def surface_light_curve(
    surface: Surface,
    r: float | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    theta: float | None = None,
    order: int = 20,
    higher_precision: bool = False,
):
    """Light curve of an occulted surface.

    Args:
        surface (Surface): Surface object
        r (float or None): radius of the occulting body, relative to the current map
           body
        x (float or None): x coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        y (float or None): y coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        z (float or None): z coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        theta (float):
            rotation angle of the map, in radians. By default 0.0
        order (int):
            order of the P integral numerical approximation. By default 20
        higher_precision (bool): whether to compute change of basis matrix as hight
            precision. By default False (only used to testing).

    Returns:
        ArrayLike: flux
    """
    if higher_precision:
        try:
            from jaxoplanet.starry.multiprecision import (
                basis as basis_mp,
                utils as utils_mp,
            )
        except ImportError as e:
            raise ImportError(
                "The `mpmath` Python package is required for higher_precision=True."
            ) from e

    total_deg = surface.deg

    rT_deg = rT(total_deg)

    x = 0.0 if x is None else x
    y = 0.0 if y is None else y
    z = 0.0 if z is None else z

    # no occulting body
    if r is None:
        b_rot = True
        theta_z = 0.0
        design_matrix_p = rT_deg

    # occulting body
    else:
        b = jnp.sqrt(jnp.square(x) + jnp.square(y))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(z, 0.0))
        b_occ = jnp.logical_not(b_rot)

        # trick to avoid nan `x=jnp.where...` grad caused by nan sT
        r = jnp.where(b_rot, 1.0, r)
        b = jnp.where(b_rot, 1.0, b)

        if surface.ydeg == 0:
            if surface.udeg == 0:
                ld_u = jnp.array([])
            else:
                ld_u = jnp.concatenate(
                    [jnp.atleast_1d(jnp.asarray(u_)) for u_ in surface.u], axis=0
                )

            lc_func = partial(_limb_dark_light_curve, ld_u, order=order)
            lc = lc_func(b, r)
            return surface.amplitude * (1.0 + jnp.where(b_occ, lc, 0))

        else:
            theta_z = jnp.arctan2(x, y)
            sT = solution_vector(total_deg, order=order)(b, r)

        if total_deg > 0:
            if higher_precision:
                A2 = np.atleast_2d(utils_mp.to_numpy(basis_mp.A2(total_deg)))
            else:
                A2 = scipy.sparse.linalg.inv(A2_inv(total_deg))
                A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
        else:
            A2 = jnp.array([[1]])

        design_matrix_p = jnp.where(b_occ, sT @ A2, rT_deg)

    if surface.ydeg == 0:
        rotated_y = surface.y.todense()
    else:
        rotated_y = left_project(
            surface.ydeg,
            surface._inc,
            surface._obl,
            theta,
            theta_z,
            surface.y.todense(),
        )

    # limb darkening
    if surface.udeg == 0:
        p_u = Pijk.from_dense(jnp.array([1]))
    else:
        u = jnp.array([1, *surface.u])
        p_u = Pijk.from_dense(u @ U(surface.udeg), degree=surface.udeg)

    # surface map * limb darkening map
    if higher_precision:
        A1_val = np.atleast_2d(utils_mp.to_numpy(basis_mp.A1(surface.ydeg)))
    else:
        A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(surface.ydeg))

    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=surface.ydeg)
    p_yu = p_y * p_u
    norm = np.pi / (p_u.tosparse() @ rT(surface.udeg))

    return surface.amplitude * (p_yu.tosparse() @ design_matrix_p) * norm

def system_observable(surface_observable, flux_planet, flux_star, **kwargs):
    _surface_observable = partial(surface_observable, **kwargs)

    def observable_fun(
        system: SurfaceSystem,
    ) -> Callable[[Scalar], tuple[Array | None, Array | None]]:
        # the observable function of the central given the position and radii
        # of all other bodies
        central_bodies_observable = jax.vmap(
            _surface_observable, in_axes=(None, 0, 0, 0, 0, None)
        )

        # the observable function of all bodies combined given their position to the
        # central
        @partial(system.surface_vmap, in_axes=(0, 0, 0, 0, None))
        def compute_body_observable(surface, radius, x, y, z, time):
            if surface is None:
                return 0.0
            else:
                theta = surface.rotational_phase(time)
                return _surface_observable(
                    surface,
                    (system.central.radius / radius),
                    (x / radius),
                    (y / radius),
                    (z / radius),
                    theta,
                )

        @partial(jnp.vectorize, signature="()->(n)")
        def observable_impl(time: Scalar) -> Array:
            # a function that give the array of observables for all bodies, starting
            # with the central
            if system.central_surface is None:
                central_light_curves = jnp.array([0.0])
            else:
                theta = system.central_surface.rotational_phase(time)
                central_radius = system.central.radius
                central_phase_curve = _surface_observable(
                    system.central_surface, theta=theta
                )
                if len(system.bodies) > 0:
                    xos, yos, zos = system.relative_position(time)
                    n = len(xos)
                    central_light_curves = central_bodies_observable(
                        system.central_surface,
                        (system.radius / central_radius),
                        (xos / central_radius),
                        (yos / central_radius),
                        (zos / central_radius),
                        theta,
                    )
                    def phase_planet(time,P,t0=0):
                        phase = jnp.sin(((time+t0)/P)*2*jnp.pi - jnp.pi/2)/2+0.5 
                        return phase
                    if n > 1 and central_light_curves is not None:
                        central_light_curves = central_light_curves.sum(
                            0
                        ) - central_phase_curve * (n - 1)
                        central_light_curves = jnp.expand_dims(central_light_curves, 0)

                    body_light_curves = compute_body_observable(
                        system.radius, -xos, -yos, -zos, time
                    )
                    in_eclipse = jnp.logical_not(body_light_curves)
                    body_light_curves_2 = (flux_planet/flux_star)*phase_planet(time,system.bodies[0].period)*(system.bodies[0].radius/system.central.radius)**2 * (-1*in_eclipse+1) 
                    #body_light_curves_2 = body_light_curves_2/jnp.max(body_light_curves_2)
                    return jnp.hstack([central_light_curves, body_light_curves_2])
                else:
                    return jnp.array([central_phase_curve])

        return observable_impl

    return observable_fun

def light_curve(system, flux_planet, flux_star, order=20):
    return system_observable(surface_light_curve, flux_planet, flux_star, order=order)(system)
    
def fast_binning(x, y, bins, error=None, std=False):
    bins = np.arange(np.min(x), np.max(x), bins)
    d = np.digitize(x, bins)

    n = np.max(d) + 2

    binned_x = np.empty(n)
    binned_y = np.empty(n)
    binned_error = np.empty(n)

    binned_x[:] = -np.pi
    binned_y[:] = -np.pi
    binned_error[:] = -np.pi

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            binned_y[i] = np.mean(y[s])
            binned_x[i] = np.mean(x[s])
            binned_error[i] = np.std(y[s]) / np.sqrt(len(s))

            if error is not None:
                err = error[s]
                binned_error[i] = np.sqrt(np.sum(np.power(err, 2))) / len(err)
            else:
                binned_error[i] = np.std(y[s]) / np.sqrt(len(s))

    nans = binned_x == -np.pi
    
    return binned_x[~nans], binned_y[~nans], binned_error[~nans]
    
df = pd.read_pickle("LHS3844b_0.0_1obs.pickle")
wvl = df['data'].keys()

df2 = pd.read_pickle("55Cnce_phase_curve_max_1obs.pickle")

P=df2['theta'][3]
R_star = df2['theta'][0]
M_star = df2['theta'][2]
time_transit = df2['theta'][4]
r = df2['theta'][6]
R = df2['theta'][6]*R_star
T_star = df2['theta'][1]
T_planet = df2['theta'][8]
uu = np.array([0.45, 0.05]) # Quadratic limb-darkening coefficients

def light_curve_model(time, lamb, R, T_planet, T_star, P, R_star, M_star, time_transit, uu): 
    star = Central(radius=R_star, mass=M_star)
    planet = Body(radius=R, time_transit=0, period=P)
    m = Surface(u=uu)
    system = SurfaceSystem(star, m).add_body(planet, m)
    flux_star = planck(lamb*1e-6, jnp.mean(T_star))
    flux_planet = planck(lamb*1e-6, jnp.mean(T_planet))
    pc = light_curve(system, flux_planet, flux_star)(time)[:,1] + light_curve(system, flux_planet, flux_star)(time)[:,0]
    return pc

instru = df2['datasets'].keys()
instru = list(instru)
instru.pop(0)

instru = df2['datasets'].keys()
instru = list(instru)
instru.pop(0)

plt.rcParams.update({
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

dataset = {}
for i, inst in enumerate(instru) : 
    time = df2['datasets'][inst]['lc']['time'] - time_transit
    pc = df2['datasets'][inst]['lc']['model']
    dataset[list(wvl)[i]] = [time, df2['datasets'][inst]['lc']['ariel'], df2['datasets'][inst]['lc']['flux_error']]
    second_binned = fast_binning(time, df2['datasets'][inst]['lc']['ariel'], 0.01, error = np.array(df2['datasets'][inst]['lc']['flux_error']))
    plt.figure()
    plt.errorbar(time, df2['datasets'][inst]['lc']['ariel'], yerr=df2['datasets'][inst]['lc']['flux_error'], c=(1, 1, 0.2), fmt='.', label='data')
    plt.errorbar(second_binned[0], second_binned[1], yerr=second_binned[2], c=(1, 0.5, 0), fmt='o', label='data binned', zorder=3)
    plt.plot(time, light_curve_model(time, list(wvl)[i], R, T_planet, T_star, P, R_star, M_star, time_transit, uu), label='our model', zorder=10)
    plt.plot(time, pc, 'k', label='origin data', zorder=5)
    plt.title(rf'Phase curve at $\lambda$ = {list(wvl)[i]} $\mu$m with {str(inst)[2:-1]}', fontsize=16)
    plt.legend(fontsize=12)
    plt.xlabel(f'BJD - {time_transit} [d]', fontsize=14)
    plt.ylabel(r'$(F_p + F_{\bigstar})/F_{\bigstar}$', fontsize=14)
    plt.savefig(f"model_55Cnce_{str(inst)[2:-1]}.svg")
    #plt.show()

def mcmc(lamb, P, R_star, M_star, time_transit, R_planet, T_star, T_planet, uu):
        time = dataset[lamb][0]
        observed = dataset[lamb][1]
        yerr_array = dataset[lamb][2]
        yerr = dataset[lamb][2][0]
        def MCMC_model(time, yerr, y=None):
            # Priors for the parameters we're fitting for

            # The radius of the planet
            r = numpyro.sample("r", numpyro.distributions.Normal(R_planet, 0.01))

            # The radius of the star
            R_star_prior = numpyro.sample("R_star", numpyro.distributions.Normal(R_star, 0.05))

            # The mass of the star
            M_star_prior = numpyro.sample("M_star", numpyro.distributions.Normal(M_star, 0.02))

            # The star temperature
            T_star_prior = numpyro.sample("T_star", numpyro.distributions.Normal(T_star, 10))

            # The planet temperature
            T_planet_prior = numpyro.sample("T_planet", numpyro.distributions.Normal(T_planet, 200))

            u1 = numpyro.sample("u1", numpyro.distributions.Normal(uu[0], 0.2))

            u2 = numpyro.sample("u2", numpyro.distributions.Normal(uu[1], 0.2))

            # The orbit and light curve
            y_pred = light_curve_model(time, lamb, r, T_planet_prior, T_star_prior, P, R_star_prior, M_star_prior, time_transit, (u1, u2))

            # Let's track the light curve
            numpyro.deterministic("light_curve", y_pred)

            # The likelihood function assuming Gaussian uncertainty
            numpyro.sample("obs", numpyro.distributions.Normal(y_pred, yerr), obs=y)

        n_prior_samples = 3000
        prior_samples = numpyro.infer.Predictive(MCMC_model, num_samples=n_prior_samples)(jax.random.PRNGKey(0), time, yerr)

        # Let's make it into an arviz InferenceData object.
        # To do so we'll first need to reshape the samples to be of shape (chains, draws, *shape)
        converted_prior_samples = {
            f"{p}": np.expand_dims(prior_samples[p], axis=0) for p in prior_samples
        }
        prior_samples_inf_data = az.from_dict(converted_prior_samples)
        fig = plt.figure(figsize=(12, 12))
        _ = corner.corner(
            prior_samples_inf_data,
            fig=fig,
            var_names=["r", "T_planet", "T_star", "M_star", "R_star", "u1", "u2"],
            truths=[R_planet, T_planet, T_star, M_star, R_star, uu[0], uu[1]],
            show_titles=True,
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 10},
        )
        plt.savefig(f"priors_55Cnce_{str(lamb)}.svg")
        init_param_method = "true_values"  # "prior_median" or "true_values"

        if init_param_method == "prior_median":
            print("Starting from the prior medians")
            run_optim = numpyro_ext.optim.optimize(
                MCMC_model, init_strategy=numpyro.infer.init_to_median()
            )
        elif init_param_method == "true_values":
            print("Starting from the true values")
            init_params = {
                #"t0": float(time_transit),
                #"logP": jnp.log(P),
                "r": R_planet,
                #"logM": jnp.log(M_star),
                "M_star" : float(M_star),
                "R_star" : float(R_star),
                "T_star" : float(T_star),
                "T_planet" : float(T_planet),
                "u1": uu[0],
                "u2": uu[1],
            }
            run_optim = numpyro_ext.optim.optimize(
                MCMC_model,
                init_strategy=numpyro.infer.init_to_value(values=init_params),
            )

        time = jnp.asarray(time, dtype=jnp.float64)
        yerr = jnp.asarray(yerr, dtype=jnp.float64)
        observed = jnp.asarray(observed, dtype=jnp.float64)

        opt_params = run_optim(jax.random.PRNGKey(3), time, yerr, y=observed)
        for k, v in opt_params.items():
            if k in ["light_curve", "obs"]:
                continue
            print(f"optimized value of {k}: {v}")
        sampler = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            MCMC_model,
            dense_mass=True,
            regularize_mass_matrix=True,
            init_strategy=numpyro.infer.init_to_value(values=opt_params),
        ),
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        progress_bar=True,
        )

        sampler.run(jax.random.PRNGKey(1), time, yerr, y=observed)
        inf_data = az.from_numpyro(sampler)
        fig = plt.figure(figsize=(12, 12))
        _ = corner.corner(
            inf_data,
            var_names=["r", "T_planet", "R_star", "M_star", "T_star", "u1", "u2"],
            truths=[R, T_planet, R_star, M_star, T_star, uu[0], uu[1]],
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 10},
            title_fmt=".10f",
            fig=fig,
        )
        plt.savefig(f"posteriors_55Cnce_{str(lamb)}.svg")
        samples = sampler.get_samples()
        r_percentiles = np.percentile(samples["r"], [16, 50, 84], axis=0)
        r_minus = r_percentiles[1]-r_percentiles[0]
        r_plus = r_percentiles[2]-r_percentiles[1]
        R_mcmc = r_percentiles[1]
        R_star_percentiles = np.percentile(samples["R_star"], [16, 50, 84], axis=0)
        R_star_minus = R_star_percentiles[1]-R_star_percentiles[0]
        R_star_plus = R_star_percentiles[2]-R_star_percentiles[1]
        R_star_mcmc = R_star_percentiles[1]
        M_star_percentiles = np.percentile(samples["M_star"], [16, 50, 84], axis=0)
        M_star_minus = M_star_percentiles[1]-M_star_percentiles[0]
        M_star_plus = M_star_percentiles[2]-M_star_percentiles[1]
        M_star_mcmc = M_star_percentiles[1]
        T_percentiles = np.percentile(samples["T_planet"], [16, 50, 84], axis=0)
        T_minus = T_percentiles[1]-T_percentiles[0]
        T_plus = T_percentiles[2]-T_percentiles[1]
        T_mcmc = T_percentiles[1]
        T_star_percentiles = np.percentile(samples["T_star"], [16, 50, 84], axis=0)
        T_star_minus = T_star_percentiles[1]-T_star_percentiles[0]
        T_star_plus = T_star_percentiles[2]-T_star_percentiles[1]
        T_star_mcmc = T_star_percentiles[1]
        u1_percentiles = np.percentile(samples["u1"], [16, 50, 84], axis=0)
        u1_minus = u1_percentiles[1]-u1_percentiles[0]
        u1_plus = u1_percentiles[2]-u1_percentiles[1]
        u1_mcmc = u1_percentiles[1]
        u2_percentiles = np.percentile(samples["u2"], [16, 50, 84], axis=0)
        u2_minus = u2_percentiles[1]-u2_percentiles[0]
        u2_plus = u2_percentiles[2]-u2_percentiles[1]
        u2_mcmc = u2_percentiles[1]

        plt.figure()
        plt.plot(time, light_curve_model(time, lamb, R_planet, T_planet, T_star, P, R_star, M_star, time_transit, uu), label='truth', zorder=3)
        plt.plot(time, light_curve_model(time, lamb, R_mcmc, T_mcmc, T_star_mcmc, P, R_star_mcmc, M_star_mcmc, time_transit, (u1_mcmc, u2_mcmc)), "--C0", label="MCMC result", zorder=3)
        plt.errorbar(time, observed, yerr=yerr_array, fmt='o', label='data')
        plt.title(rf'Phase curve at $\lambda$ = {lamb} $\mu$m')
        plt.legend()
        plt.xlabel(r'BJD$_{TDB}$ - 2,458,829 [d]')
        plt.ylabel(r'$\frac{F_p + F_{\star}}{F_{\star}}$')
        #plt.show()
        data = jnp.array([[r_minus, r_plus, R_mcmc], [T_minus, T_plus, T_mcmc], [T_star_minus, T_star_plus, T_star_mcmc], [R_star_minus, R_star_plus, R_star_mcmc], [M_star_minus, M_star_plus, M_star_mcmc],
                          [u1_minus, u1_plus, u1_mcmc], [u2_minus, u2_plus, u2_mcmc]])
        np.savetxt(f"result_mcmc_55Cnce_{lamb}.txt", data, fmt="%.2f", delimiter="\t", header="16th percentile\t84th percentile\t50th percentile")

        return data
        
for i, inst in enumerate(instru) : 
    data = mcmc(list(wvl)[i], P, R_star, M_star, time_transit, R, T_star, T_planet, uu)


