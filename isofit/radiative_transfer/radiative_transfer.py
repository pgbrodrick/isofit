#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
#          Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#          Jay E. Fahlen, jay.e.fahlen@jpl.nasa.gov
#
from __future__ import annotations

import logging

import numpy as np

from isofit.core import units
from isofit.core.common import eps, svd_inv_sqrt
from isofit.radiative_transfer.engines import Engines

Logger = logging.getLogger(__file__)


def confPriority(key, configs):
    """
    Selects a key from a config if the value for that key is not None
    Prioritizes returning the first value found in the configs list

    TODO: ISOFIT configs are annoying and will create keys to NoneTypes
    Should use mlky to handle key discovery at runtime instead of like this
    """
    value = None
    for config in configs:
        if hasattr(config, key):
            value = getattr(config, key)
            if value is not None:
                break
    return value


class RadiativeTransfer:
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and
    TIR. This class maintains the master list of statevectors.
    """

    # Keys to retrieve from 3 sections to use the preferred
    # Prioritizes retrieving from radiative_transfer_engines first, then instrument, then radiative_transfer
    _keys = [
        "interpolator_style",
        "overwrite_interpolator",
        "lut_grid",
        "lut_path",
        "wavelength_file",
    ]

    def __init__(self, full_config: Config):
        config = full_config.forward_model.radiative_transfer
        confIT = full_config.forward_model.instrument

        self.lut_grid = config.lut_grid
        self.statevec_names = config.statevector.get_element_names()
        self.terrain_style = config.terrain_style
        self.min_cos_i = config.min_cos_i

        self.rt_engines = []
        for idx in range(len(config.radiative_transfer_engines)):
            confRT = config.radiative_transfer_engines[idx]

            if confRT.engine_name not in Engines:
                raise AttributeError(
                    f"Invalid radiative transfer engine choice. Got: {confRT.engine_name}; Must be one of: {list(Engines)}"
                )

            # Generate the params for this RTE
            params = {
                key: confPriority(key, [confRT, confIT, config]) for key in self._keys
            }
            params["engine_config"] = confRT
            params["n_cores"] = full_config.implementation.n_cores

            # Select the right RTE and initialize it
            rte = Engines[confRT.engine_name](**params)
            self.rt_engines.append(rte)

            # Make sure the length of the config statevectores match the engine's assumed statevectors
            if (expected := len(config.statevector.get_element_names())) != (
                got := len(rte.indices.x_RT)
            ):
                error = f"Mismatch between the number of elements for the config statevector and LUT.indices.x_RT: {expected=}, {got=}"
                Logger.error(error)
                raise AttributeError(error)

        # The rest of the code relies on sorted order of the individual RT engines which cannot
        # be guaranteed by the dict JSON or YAML input
        self.rt_engines.sort(key=lambda x: x.wl[0])

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv, sv_name in zip(*config.statevector.get_elements()):
            self.bounds.append(sv.bounds)
            self.scale.append(sv.scale)
            self.init.append(sv.init)
            self.prior_sigma.append(sv.prior_sigma)
            self.prior_mean.append(sv.prior_mean)

        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)
        self.Sa_cached = np.diagflat(np.power(self.prior_sigma, 2))
        self.Sa_normalized = self.Sa_cached / np.mean(np.diag(self.Sa_cached))
        self.Sa_inv_normalized, self.Sa_inv_sqrt_normalized = svd_inv_sqrt(
            self.Sa_normalized
        )

        self.wl = np.concatenate([RT.wl for RT in self.rt_engines])

        self.bvec = config.unknowns.get_element_names()
        self.bval = np.array([x for x in config.unknowns.get_elements()[0]])

        self.solar_irr = np.concatenate([RT.solar_irr for RT in self.rt_engines])

    def xa(self):
        """Pull the priors from each of the individual RTs."""
        return self.prior_mean

    def Sa(self):
        """Pull the priors from each of the individual RTs."""
        return self.Sa_cached

    def Sb(self):
        """Uncertainty due to unmodeled variables."""
        return np.diagflat(np.power(self.bval, 2))

    def get_shared_rtm_quantities(self, x_RT, geom):
        """Return only the set of RTM quantities (transup, sphalb, etc.) that are contained
        in all RT engines.
        """
        ret = []
        for RT in self.rt_engines:
            ret.append(RT.get(x_RT, geom))

        return self.pack_arrays(ret)

    @property
    def coszen(self):
        """
        Backwards compatibility until Geometry takes over this param
        Return some child RTE coszen
        """
        for child in self.rt_engines:
            if "coszen" in child.lut:
                return child.lut.coszen.data

    def calc_rdn(
        self,
        x_RT,
        rho_dir_dir,
        rho_dif_dir,
        Ls,
        L_tot,
        L_dir_dir,
        L_dif_dir,
        L_dir_dif,
        L_dif_dif,
        r,
        geom,
    ):
        """
        Physics-based forward model to calculate at-sensor radiance.
        Includes topography, background reflectance, and glint.
        """
        # Adjacency effects
        # ToDo: we need to think about if we want to obtain the background reflectance from the Geometry object
        #  or from the surface model, i.e., the same way as we do with the target pixel reflectance

        rho_dir_dif = (
            geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dir_dir
        )
        rho_dif_dif = (
            geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dif_dir
        )

        # Atmospheric path radiance
        L_atm = self.get_L_atm(x_RT, geom)

        # Atmospheric spherical albedo
        s_alb = r["sphalb"]
        atm_surface_scattering = s_alb * rho_dif_dif
        eq_11_term = 1 - atm_surface_scattering

        # Special case: 1-component model
        if not isinstance(L_dir_dir, np.ndarray) or len(L_dir_dir) == 1:
            # we assume rho_dir_dir = rho_dif_dir = rho_dir_dif = rho_dif_dif
            rho_dif_dif = rho_dir_dir
            # eliminate spherical albedo and one reflectance term from numerator if using 1-component model
            atm_surface_scattering = 1
            eq_11_term = 1

        # Thermal transmittance
        L_up = Ls * self.get_upward_transm(r=r, geom=geom)

        # Our radiance model follows the physics as presented in Guanter (2006), Vermote et al. (1997), and
        # Tanre et al. (1983). This particular formulation facilitates the consideration of topographic effects,
        # glint, or BRDF modeling in general. The contribution of the target to the signal at the top of the atmosphere
        # is decomposed as the sum of four terms:

        # 1. photons directly transmitted from the sun to the target and directly reflected back to the sensor
        #    rho_dir_dir => directional-directional surface reflectance of the target
        # 2. photons scattered by the atmosphere then reflected by the target and directly transmitted to the sensor
        #    rho_dif_dir => surface diffuse-directional reflectance
        # 3. photons directly transmitted to the target but scattered by the atmosphere on their way to the sensor
        #    rho_dir_dif => surface directional-diffuse reflectance
        # 4. photons having at least two interactions with the atmosphere and one with the target
        #    rho_dif_dif => surface diffuse-diffuse reflectance

        # These terms are also called coupling terms, as they are responsible for the coupling between atmospheric
        # radiative transfer and the surface reflectance properties.

        # The coupling terms are multiplied by four different combinations of direct and diffuse radiance terms:
        # 1. L_dir_dir => downward direct * upward direct
        # 2. L_dif_dir => downward diffuse * upward direct
        # 3. L_dir_dif => downward direct * upward diffuse
        # 4. L_dif_dif => downward diffuse * upward diffuse

        # When separated radiance terms and/or a BRDF model of the surface are not available,
        # the Lambertian assumption is made for the target reflectance:
        # rho_dir_dir = rho_dif_dir = rho_dir_dif = rho_dif_dif
        # In this case, our radiance model reduces to:
        # L_atm + (L_tot * rho_dir_dir) / (1 - S * rho_dir_dir) + L_up,
        # with L_tot being the total radiance (downward * upward, direct + diffuse).

        # TOA radiance model
        ret = (
            L_atm
            + L_dir_dir * rho_dir_dir
            + L_dif_dir * rho_dif_dir / eq_11_term
            + L_dir_dif * rho_dir_dif
            + L_dif_dif * rho_dif_dif / eq_11_term
            + (L_tot * atm_surface_scattering * rho_dif_dif) / (1 - s_alb * rho_dif_dif)
            + L_up
        )

        return ret

    def get_L_atm(self, x_RT: np.array, geom: Geometry) -> np.array:
        """Get the interpolated modeled atmospheric path radiance.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated modeled atmospheric path radiance
        """
        L_atms = []

        verified_geom = geom.verify(self.coszen)
        coszen, cos_i = verified_geom["coszen"], verified_geom["cos_i"]

        for RT in self.rt_engines:
            if RT.treat_as_emissive:
                r = RT.get(x_RT, geom)
                rdn = r["thermal_upwelling"]
                L_atms.append(rdn)
            else:
                r = RT.get(x_RT, geom)
                if RT.rt_mode == "rdn":
                    L_atm = r["rhoatm"]
                else:
                    rho_atm = r["rhoatm"]
                    L_atm = units.transm_to_rdn(rho_atm, coszen, self.solar_irr)
                L_atms.append(L_atm)
        return np.hstack(L_atms)

    def get_L_coupled(self, r: dict, geom: Geometry):
        """Get the interpolated radiance terms on the sun-to-surface-to-sensor path.
        These follow the physics as presented in Guanter (2006), Vermote et al. (1997), and Tanre et al. (1983).

        Args:
            r:      interpolated radiative transfer quantities from the LUT
            coszen: top-of-atmosphere solar zenith angle
            cos_i:  local solar zenith angle at the surface

        Returns:
            interpolated radiances along all optical paths:
            L_dir_dir => downward direct * upward direct
            L_dif_dir => downward diffuse * upward direct
            L_dir_dif => downward direct * upward diffuse
            L_dif_dif => downward diffuse * upward diffuse
        """
        # Check coszen against cos_i
        verified_geom = geom.verify(self.coszen)
        coszen, cos_i, skyview_factor = (
            verified_geom["coszen"],
            verified_geom["cos_i"],
            verified_geom["skyview_factor"],
        )
        # Pretend that the surface is flat, regardless of input geometry
        if self.terrain_style == "flat":
            cos_i = coszen

        cos_i = max(self.min_cos_i, cos_i)

        # radiances along all optical paths
        L_coupled = []

        if any(
            [
                not isinstance(r[key], np.ndarray) or len(r[key]) == 1
                for key in self.rt_engines[0].coupling_terms
            ]
        ):
            # In case of the 1-component model, we cannot populate the coupling terms
            L_coupled = [
                0,
                0,
                0,
                0,
            ]
        else:
            for key in self.rt_engines[0].coupling_terms:
                L_coupled.append(
                    units.transm_to_rdn(r[key], coszen=coszen, solar_irr=self.solar_irr)
                    if self.rt_engines[0].rt_mode == "transm"
                    else r[key]
                )
        # Topographic shadow mask (0=shadow, 1=sunlit pixel).
        # for now, this is always set to 1.0.
        b = 1.0

        # Assigning coupled terms, unscaling and rescaling downward direct radiance by local solar zenith angle.
        # Downward diffuse components are scaled by viewable sky fraction (i.e., "ungula" of viewable sky in solid geometry terms).
        L_dir_dir = L_coupled[0] / coszen * cos_i * b
        L_dif_dir = L_coupled[1]
        L_dir_dif = L_coupled[2] / coszen * cos_i * b
        L_dif_dif = L_coupled[3]

        # Note - we should really be doing the multiplication upstream before convolution - this is an approximation
        # Correct downward diffuse term for topographic assuming Hay's model (Hay 1979; Richter 1998; Guanter et al., 2009)
        t_down_dir = r["transm_down_dir"]
        hays_model = (b * t_down_dir * (cos_i / coszen)) + (
            (1 - b * t_down_dir) * skyview_factor
        )
        # applies to the downward diffuse terms
        L_dif_dir *= hays_model
        L_dif_dif *= hays_model

        return L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif

    def calc_RT_quantities(self, x_RT: np.ndarray, geom: Geometry):
        """Retrieves the RT quantities including the LUT sample (r),
        and the radiances (L). This function handles the hand-off between
        the 1c and 4c model.

        In the 1c case, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = 0,
        and L_tot, L_down_dir, and L_down_dif are populated within the
        if statement.

        In the 4c case, we always use returns from get_L_coupled

        All quantities are on the sun-to-surface-to-sensor path.

        """

        # Propogate LUT
        r = self.get_shared_rtm_quantities(x_RT, geom)

        # Default: get directional radiances
        L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = self.get_L_coupled(r, geom)
        L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif

        # Handle 1c L_tot. NOTE: transm_down_dif = total transm for 1c case.
        if not isinstance(L_tot, np.ndarray) or len(L_tot) == 1:
            coszen = geom.verify(self.coszen)["coszen"]
            L_tots = []
            for RT in self.rt_engines:
                r = RT.get(x_RT, geom)
                if RT.treat_as_emissive:
                    rdn = r["thermal_downwelling"]
                    L_tots.append(rdn)
                else:
                    if RT.rt_mode == "rdn":
                        L_tot = r["transm_down_dif"]
                    else:
                        L_tot = units.transm_to_rdn(
                            r["transm_down_dif"],
                            coszen,
                            self.solar_irr,
                        )
                    L_tots.append(L_tot)
            L_tot = np.hstack(L_tots)

        return (
            r,
            L_tot,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        )

    def get_upward_transm(self, r: dict, geom: Geometry, max_transm: float = 1.05):
        """
        Get total upward transmittance w/physical check enforced (max_transm) and hand-off between 1c and 4c model.

        This is called for all surfaces to handle thermal downwelling/upwelling component.
        While rt can be either rdn or transm modes, this must be in units of transmittance.

        """
        transm_up_dir = r["transm_up_dir"]
        transm_up_dif = r["transm_up_dif"]

        # NOTE for 1c case transm-up is not a key, and therefore Ls and transup is zero.
        if not isinstance(transm_up_dir, np.ndarray) or len(transm_up_dir) == 1:
            return np.zeros_like(self.solar_irr, dtype=np.float32)
        else:
            transup = transm_up_dir + transm_up_dif

            if np.max(transup) > max_transm:
                raise ValueError(
                    (
                        f"Upward transmittance (max:{np.max(transup)}) is greater than {max_transm}. "
                        f"Verify 'transm_up_dir' and 'transm_up_dif' keys are in units of transmittance."
                    )
                )
            return transup

    def drdn_dRT(self, x_RT, geom, rho_dir_dir, rho_dif_dir, Ls, rdn, fd=False):
        """Derivative of estimated radiance w.r.t. RT statevector elements.
        We use a numerical approach to approximate dRT with a constant surface
        reflectance. This is a reasonable approx. for the multicomponent surface.

        When using the glint model however, this does not take into account
        the dependence of the surface reflectance on the atmosphere.
        """
        K_RT = []
        for RT in self.rt_engines:
            # perturb each element of the RT state vector (finite difference)
            # do this if flag is set, or we're in transmission mode
            # below we actually do transmission mode cases....but I feel like
            # that's probably really messy
            if fd or RT.rt_mode == "transm":
                x_RTs_perturb = x_RT + np.eye(len(x_RT)) * eps
                for x_RT_perturb in list(x_RTs_perturb):
                    (
                        r,
                        L_tot,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                    ) = self.calc_RT_quantities(x_RT_perturb, geom)

                    # Surface state is held constant?
                    rdne = self.calc_rdn(
                        x_RT_perturb,
                        rho_dir_dir,
                        rho_dif_dir,
                        Ls,
                        L_tot,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                        r,
                        geom,
                    )
                    K_RT.append((rdne - rdn) / eps)

                # K_RT = np.array(K_RT).T
            else:
                # Analytical derivative using the VectorInterpolator derivative method

                # Get the point for interpolation
                point = np.zeros(RT.n_point)
                point[RT.indices.x_RT] = x_RT
                for i, key in RT.indices.geom.items():
                    point[i] = getattr(geom, key)

                if RT.indices.convert_observer_zenith:
                    point[RT.indices.convert_observer_zenith] = (
                        180.0 - point[RT.indices.convert_observer_zenith]
                    )

                # Get the derivatives of the RT quantities w.r.t. the state vector
                dr_dx = {}
                for key, lut in RT.luts.items():
                    dr_dx[key] = lut.derivative(point)[RT.indices.x_RT, :]

                # Now we need to calculate drdn/dr_i for each RT quantity
                # This requires differentiating the calc_rdn equation w.r.t. each RT quantity

                # Get the base quantities
                r, L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = (
                    self.calc_RT_quantities(x_RT, geom)
                )

                # Special Terms
                rho_dir_dif = (
                    geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dir_dir
                )
                rho_dif_dif = (
                    geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dif_dir
                )

                s_alb = r["sphalb"]
                atm_surface_scattering = s_alb * rho_dif_dif
                eq_11_term = 1 - atm_surface_scattering

                # 3c model
                if not isinstance(L_dir_dir, np.ndarray) or len(L_dir_dir) == 1:
                    rho_dif_dif = rho_dir_dir
                    atm_surface_scattering = 1
                    eq_11_term = 1

                # Calculate the derivative of the radiance w.r.t. each RT state vector element
                # using the chain rule: drdn/dx_RT = sum_i (drdn/dr_i * dr_i/dx_RT)

                # Initialize the derivative array
                drdn_dx = np.zeros((len(RT.indices.x_RT), len(self.wl)))

                # 1. Derivative for L_atm
                if RT.treat_as_emissive:
                    dL_atm_dx = dr_dx["thermal_upwelling"]
                else:
                    if RT.rt_mode == "rdn":
                        dL_atm_dx = dr_dx["rhoatm"]
                    else:
                        verified_geom = geom.verify(self.coszen)
                        coszen = verified_geom["coszen"]
                        dL_atm_dx = units.transm_to_rdn(
                            dr_dx["rhoatm"], coszen, self.solar_irr
                        )
                drdn_dx += dL_atm_dx

                # 2. Derivative for L_up
                # L_up = Ls * (transm_up_dir + transm_up_dif)
                if (
                    isinstance(r["transm_up_dir"], np.ndarray)
                    and len(r["transm_up_dir"]) > 1
                ):
                    dtransup_dx = dr_dx["transm_up_dir"] + dr_dx["transm_up_dif"]
                    dL_up_dx = Ls * dtransup_dx
                    drdn_dx += dL_up_dx

                # 3. Derivative for coupling terms
                if not (not isinstance(L_dir_dir, np.ndarray) or len(L_dir_dir) == 1):
                    verified_geom = geom.verify(self.coszen)
                    coszen, cos_i, skyview_factor = (
                        verified_geom["coszen"],
                        verified_geom["cos_i"],
                        verified_geom["skyview_factor"],
                    )
                    if self.terrain_style == "flat":
                        cos_i = coszen
                    cos_i = max(self.min_cos_i, cos_i)
                    b = 1.0

                    # Get derivatives of coupling terms
                    dL_coupled_dx = []
                    for key in RT.coupling_terms:
                        if RT.rt_mode == "transm":
                            dL_coupled_dx.append(
                                units.transm_to_rdn(
                                    dr_dx[key], coszen=coszen, solar_irr=self.solar_irr
                                )
                            )
                        else:
                            dL_coupled_dx.append(dr_dx[key])

                    dL_dir_dir_dx = dL_coupled_dx[0] / coszen * cos_i * b
                    dL_dif_dir_dx = dL_coupled_dx[1]
                    dL_dir_dif_dx = dL_coupled_dx[2] / coszen * cos_i * b
                    dL_dif_dif_dx = dL_coupled_dx[3]

                    # Hay's model correction
                    t_down_dir = r["transm_down_dir"]
                    dt_down_dir_dx = dr_dx["transm_down_dir"]

                    hays_model = (b * t_down_dir * (cos_i / coszen)) + (
                        (1 - b * t_down_dir) * skyview_factor
                    )
                    dhays_model_dx = (b * dt_down_dir_dx * (cos_i / coszen)) - (
                        b * dt_down_dir_dx * skyview_factor
                    )

                    # Product rule for L_dif_dir and L_dif_dif
                    # L_dif_dir_corrected = L_dif_dir * hays_model
                    # d(L_dif_dir_corrected)/dx = dL_dif_dir/dx * hays_model + L_dif_dir * dhays_model_dx
                    dL_dif_dir_corrected_dx = (
                        dL_dif_dir_dx * hays_model + L_dif_dir * dhays_model_dx
                    )
                    dL_dif_dif_corrected_dx = (
                        dL_dif_dif_dx * hays_model + L_dif_dif * dhays_model_dx
                    )

                    # Add to total derivative
                    drdn_dx += dL_dir_dir_dx * rho_dir_dir
                    drdn_dx += dL_dif_dir_corrected_dx * rho_dif_dir / eq_11_term
                    drdn_dx += dL_dir_dif_dx * rho_dir_dif
                    drdn_dx += dL_dif_dif_corrected_dx * rho_dif_dif / eq_11_term

                    # 4. Derivative for spherical albedo
                    # The s_alb term appears in eq_11_term = 1 - s_alb * rho_dif_dif
                    # and in the denominator of the L_tot term: (1 - s_alb * rho_dif_dif)
                    ds_alb_dx = dr_dx["sphalb"]

                    # Derivative of (L_dif_dir * rho_dif_dir / eq_11_term) for s_alb
                    # d/dx (A / (1 - s_alb * B)) = A * B * ds_alb_dx / (1 - s_alb * B)^2
                    term1 = (
                        (L_dif_dir * rho_dif_dir)
                        * rho_dif_dif
                        * ds_alb_dx
                        / (eq_11_term**2)
                    )
                    drdn_dx += term1

                    # Derivative of (L_dif_dif * rho_dif_dif / eq_11_term) for s_alb
                    term2 = (
                        (L_dif_dif * rho_dif_dif)
                        * rho_dif_dif
                        * ds_alb_dx
                        / (eq_11_term**2)
                    )
                    drdn_dx += term2

                    # 5. Derivative of the L_tot term
                    # L_tot_term = (L_tot * s_alb * rho_dif_dif * rho_dif_dif) / (1 - s_alb * rho_dif_dif)
                    # Let L_tot = L_dir_dir + L_dif_dir_corrected + L_dir_dif + L_dif_dif_corrected
                    dL_tot_dx = (
                        dL_dir_dir_dx
                        + dL_dif_dir_corrected_dx
                        + dL_dir_dif_dx
                        + dL_dif_dif_corrected_dx
                    )

                    # Quotient rule for L_tot_term
                    # u = L_tot * s_alb * rho_dif_dif * rho_dif_dif
                    # v = 1 - s_alb * rho_dif_dif
                    # du/dx = dL_tot_dx * s_alb * rho_dif_dif^2 + L_tot * ds_alb_dx * rho_dif_dif^2
                    # dv/dx = -ds_alb_dx * rho_dif_dif
                    # d(u/v)/dx = (du/dx * v - u * dv/dx) / v^2

                    u = L_tot * s_alb * (rho_dif_dif**2)
                    v = eq_11_term
                    du_dx = dL_tot_dx * s_alb * (
                        rho_dif_dif**2
                    ) + L_tot * ds_alb_dx * (rho_dif_dif**2)
                    dv_dx = -ds_alb_dx * rho_dif_dif

                    dL_tot_term_dx = (du_dx * v - u * dv_dx) / (v**2)
                    drdn_dx += dL_tot_term_dx
                else:
                    # 1c case
                    # L_tot_term = L_tot * rho_dir_dir
                    # L_tot = transm_down_dif (or thermal_downwelling)
                    if RT.treat_as_emissive:
                        dL_tot_dx = dr_dx["thermal_downwelling"]
                    else:
                        if RT.rt_mode == "rdn":
                            dL_tot_dx = dr_dx["transm_down_dif"]
                        else:
                            verified_geom = geom.verify(self.coszen)
                            coszen = verified_geom["coszen"]
                            dL_tot_dx = units.transm_to_rdn(
                                dr_dx["transm_down_dif"], coszen, self.solar_irr
                            )

                    drdn_dx += dL_tot_dx * rho_dir_dir

                K_RT.append(drdn_dx)

        K_RT = np.array(K_RT).T
        return K_RT

    def drdn_dRTb(self, x_RT, geom, rho_dir_dir, rho_dif_dir, Ls, rdn):
        """Derivative of estimated rdn w.r.t. H2O_ABSCO

        Currently, the K_b matrix only covers forward model derivatives
        due to H2O_ABSCO unknowns, so that subsequent errors might occur
        when water vapor is not part of the statevector
        (which is very unlikely though).
        """
        if len(self.bvec) == 0:
            Kb_RT = np.zeros((0, len(self.wl.shape)))

        # ToDo: might require modification in case more unknowns are added
        # The following statement captures the case that H2O is not part
        # of the statevector.
        # but might need to be modified as soon as we add more unknowns
        elif len(self.bvec) > 0 and "H2OSTR" not in self.statevec_names:
            Kb_RT = np.zeros((1, len(self.wl)))
        else:
            # unknown parameters modeled as random variables per
            # Rodgers et al (2000) K_b matrix.  We calculate these derivatives
            # by finite differences
            Kb_RT = []
            perturb = 1.0 + eps
            for unknown in self.bvec:
                if unknown == "H2O_ABSCO" and "H2OSTR" in self.statevec_names:
                    i = self.statevec_names.index("H2OSTR")
                    x_RT_perturb = x_RT.copy()
                    x_RT_perturb[i] = x_RT[i] * perturb
                    (
                        r,
                        L_tot,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                    ) = self.calc_RT_quantities(x_RT_perturb, geom)

                    rdne = self.calc_rdn(
                        x_RT_perturb,
                        rho_dir_dir,
                        rho_dif_dir,
                        Ls,
                        L_tot,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                        r,
                        geom,
                    )
                    Kb_RT.append((rdne - rdn) / eps)

        Kb_RT = np.array(Kb_RT).T
        return Kb_RT

    def summarize(self, x_RT, geom):
        ret = []
        for RT in self.rt_engines:
            ret.append(RT.summarize(x_RT, geom))
        ret = "\n".join(ret)
        return ret

    def pack_arrays(self, rtm_quantities_from_RT_engines):
        """Take the list of dict outputs from each RT engine and
        stack their internal arrays in the same order. Keep only
        those quantities that are common to all RT engines.
        """
        # Get the intersection of the sets of keys from each of the rtm_quantities_from_RT_engines
        shared_rtm_keys = set(rtm_quantities_from_RT_engines[0].keys())
        if len(rtm_quantities_from_RT_engines) > 1:
            for rtm_quantities_from_one_RT_engine in rtm_quantities_from_RT_engines[1:]:
                shared_rtm_keys.intersection_update(
                    rtm_quantities_from_one_RT_engine.keys()
                )

        # Concatenate the different band ranges
        rtm_quantities_concatenated_over_RT_bands = {}
        for key in shared_rtm_keys:
            temp = [x[key] for x in rtm_quantities_from_RT_engines]
            rtm_quantities_concatenated_over_RT_bands[key] = np.hstack(temp)

        return rtm_quantities_concatenated_over_RT_bands


def ext550_to_vis(ext550):
    """VIS is defined as a function of the surface aerosol extinction coefficient
    at 550 nm in km-1, EXT550, by the formula VIS[km] = ln(50) / (EXT550 + 0.01159),
    where 0.01159 is the surface Rayleigh scattering coefficient at 550 nm in km-1
    (see MODTRAN6 manual, p. 50).
    """
    return np.log(50.0) / (ext550 + 0.01159)
