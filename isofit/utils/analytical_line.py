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
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
from __future__ import annotations

import logging
import multiprocessing
import os
import time
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import click
import numpy as np
from spectral.io import envi

from isofit import ray
from isofit.configs import configs
from isofit.core.common import envi_header, load_esd, load_spectrum, load_wavelen
from isofit.core.fileio import initialize_output, write_bil_chunk
from isofit.core.forward import ForwardModel
from isofit.core.geometry import Geometry
from isofit.core.multistate import (
    construct_full_state,
    fill_statevector,
    index_spectra_by_surface,
    update_config_for_surface,
)
from isofit.inversion.inverse_simple import (
    invert_algebraic,
    invert_analytical,
    invert_simple,
)
from isofit.utils.atm_interpolation import atm_interpolation


def batch_create_geometries(
    obs_array: np.ndarray,
    loc_array: np.ndarray,
    svf_array: np.ndarray,
    esd: float,
    coszen: float,
    full_config,
) -> list[Geometry]:
    """Create multiple Geometry objects at once.

    Args:
        obs_array: (n_pixels, n_obs_bands) observation data
        loc_array: (n_pixels, n_loc_bands) location data
        svf_array: (n_pixels,) skyview factor data
        esd: Earth-sun distance
        coszen: Cosine of zenith angle
        full_config: ISOFIT configuration object

    Returns:
        List of Geometry objects
    """
    n_pixels = obs_array.shape[0]
    geometries = []

    for i in range(n_pixels):
        geom = Geometry(
            obs=obs_array[i, :],
            loc=loc_array[i, :],
            esd=esd,
            svf=svf_array[i] if len(svf_array) > 0 else 1,
            coszen=coszen,
            full_config=full_config,
        )
        geometries.append(geom)

    return geometries


def batch_invert_analytical(
    fm,
    winidx: np.ndarray,
    meas_batch: np.ndarray,
    geom_batch: list[Geometry],
    x0_batch: np.ndarray,
    sub_state_batch: np.ndarray,
    num_iter: int = 1,
    diag_uncert: bool = True,
    outside_ret_const: float = -0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform analytical inversion on a batch of pixels.

    This function processes multiple pixels that share the same atmospheric state,
    allowing for more efficient computation through vectorization.

    Args:
        fm: Forward model
        winidx: Indices of retrieval windows
        meas_batch: (n_pixels, n_channels) radiance measurements
        geom_batch: List of n_pixels Geometry objects
        x0_batch: (n_pixels, n_state) initial state vectors
        sub_state_batch: (n_pixels, n_state) superpixel state vectors
        num_iter: Number of iterations
        diag_uncert: Whether to return diagonal uncertainty
        outside_ret_const: Constant value for bands outside retrieval windows

    Returns:
        trajectories: (n_pixels, num_iter+1, n_state) state trajectories
        uncertainties: (n_pixels, n_state) posterior uncertainties
    """
    n_pixels = meas_batch.shape[0]
    n_state = x0_batch.shape[1]
    n_channels = meas_batch.shape[1]

    trajectories = np.zeros((n_pixels, num_iter + 1, n_state))
    uncertainties = np.zeros((n_pixels, n_state))

    trajectories[:, 0, :] = x0_batch

    # Process each pixel (vectorization possible for shared atmosphere)
    for px in range(n_pixels):
        x = x0_batch[px].copy()
        sub_state = sub_state_batch[px]
        geom = geom_batch[px]
        meas = meas_batch[px]

        x_surface, x_atmosphere, x_instrument = fm.unpack(x)
        sub_surface, sub_atmosphere, sub_instrument = fm.unpack(sub_state)

        # Surface reflectance at RT resolution
        rho_dir_dir, rho_dif_dir = fm.calc_rfl(sub_state, geom)
        rho_dif_dir = fm.upsample(fm.surface.wl, rho_dif_dir)

        rho_dif_dif = (
            fm.upsample(fm.surface.wl, geom.bg_rfl)
            if isinstance(geom.bg_rfl, np.ndarray)
            else rho_dif_dir
        )

        # Atmosphere quantities
        (
            r,
            L_tot,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        ) = fm.calc_atmosphere_quantities(x_atmosphere, geom, rho_dif_dif=rho_dif_dif)

        L_atm = fm.atmosphere.get_L_atm(x_atmosphere, geom)
        s = r["sphalb"]
        bg = s * rho_dif_dir
        eof_offset = fm.eof_offset(sub_instrument)

        full_idx = np.concatenate((winidx, fm.idx_surf_nonrfl), axis=0)
        outside_ret_windows = np.ones(len(fm.idx_surface), dtype=bool)
        outside_ret_windows[full_idx] = False
        outside_ret_windows = np.where(outside_ret_windows)[0]
        iv_idx = fm.surface.analytical_iv_idx

        # H matrix
        H = fm.surface.analytical_model(
            bg,
            L_tot=L_tot,
            geom=geom,
            L_dir_dir=L_dir_dir,
            L_dir_dif=L_dir_dif,
            L_dif_dir=L_dif_dir,
            L_dif_dif=L_dif_dif,
        )
        L = H[winidx, :][:, iv_idx]

        # Iterate
        for n in range(num_iter):
            Seps = fm.Seps(x, meas, geom)[winidx, :][:, winidx]
            Sa, Sa_inv, Sa_inv_sqrt = fm.Sa(x, geom)
            Sa_inv = Sa_inv[fm.idx_surface, :][:, fm.idx_surface]

            xa_full = fm.xa(x, geom)
            xa_surface = xa_full[fm.idx_surface]
            prprod = Sa_inv @ xa_surface

            x_surface, x_atmosphere, x_instrument = fm.unpack(x)

            C = dpotrf(Seps, 1)[0]
            P = dpotri(C, 1)[0]

            P_tilde = ((L.T @ P) @ L).T
            P_rcond = Sa_inv[iv_idx, :][:, iv_idx] + P_tilde

            LI_rcond = dpotrf(P_rcond)[0]
            C_rcond = dpotri(LI_rcond)[0]

            y = meas[winidx] - L_atm[winidx] - eof_offset[winidx]
            xk = dsymv(1, C_rcond, (L.T @ dsymv(1, P, y) + prprod[iv_idx]))

            x_surface[iv_idx] = xk
            if outside_ret_const is None:
                x_surface[outside_ret_windows] = xa_surface[outside_ret_windows]
            else:
                x_surface[outside_ret_windows] = outside_ret_const

            x[fm.idx_surface] = x_surface
            trajectories[px, n + 1, :] = x

        if diag_uncert:
            if len(C_rcond):
                full_unc = np.ones(len(x))
                full_unc[iv_idx] = np.sqrt(np.diag(C_rcond))
            else:
                full_unc = np.ones(len(x))
                full_unc[iv_idx] = -9999

            uncertainties[px, :] = full_unc

    return trajectories, uncertainties


def batch_read_pixels(
    index_pairs: np.ndarray,
    rdn_memmap: np.ndarray,
    loc_memmap: np.ndarray,
    obs_memmap: np.ndarray,
    rt_state_memmap: np.ndarray,
    svf_memmap: np.ndarray,
    subs_state_memmap: np.ndarray,
    lbl_memmap: np.ndarray,
    iv_idx: np.ndarray,
    idx_atmosphere: np.ndarray,
    idx_instrument: np.ndarray,
    nstate: int,
    radiance_correction: np.ndarray = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list,
]:
    """Read a batch of pixels from memory-mapped arrays efficiently.

    Args:
        index_pairs: (n_pixels, 2) array of (row, col) indices
        rdn_memmap: Radiance memory map
        loc_memmap: Location memory map
        obs_memmap: Observation memory map
        rt_state_memmap: RT state memory map
        svf_memmap: Skyview factor memory map
        subs_state_memmap: Subsample state memory map
        lbl_memmap: Label memory map
        iv_idx: Analytical inversion indices
        idx_atmosphere: Atmosphere state indices
        idx_instrument: Instrument state indices
        nstate: Number of state vector elements
        radiance_correction: Optional radiance correction factor

    Returns:
        Tuple of (meas_batch, loc_batch, obs_batch, svf_batch, x_atmosphere_batch, sub_state_batch, lbl_idx_batch, valid_mask)
    """
    n_pixels = len(index_pairs)
    n_channels = rdn_memmap.shape[2]

    # Pre-allocate arrays
    meas_batch = np.zeros((n_pixels, n_channels))
    loc_batch = np.zeros((n_pixels, loc_memmap.shape[2]))
    obs_batch = np.zeros((n_pixels, obs_memmap.shape[2]))
    svf_batch = np.zeros(n_pixels) if len(svf_memmap) > 0 else np.array([])
    x_atmosphere_batch = np.zeros((n_pixels, len(idx_atmosphere)))
    sub_state_batch = np.zeros((n_pixels, nstate))
    lbl_idx_batch = np.zeros(n_pixels, dtype=int)
    valid_mask = []

    for i, (r, c, *_) in enumerate(index_pairs):
        meas = rdn_memmap[r, c, :]

        if radiance_correction is not None:
            meas = meas.copy() * radiance_correction

        if np.all(meas < 0):
            continue

        meas_batch[i, :] = meas
        loc_batch[i, :] = loc_memmap[r, c, :]
        obs_batch[i, :] = obs_memmap[r, c, :]
        if len(svf_memmap) > 0:
            svf_batch[i] = svf_memmap[r, c]
        x_atmosphere_batch[i, :] = rt_state_memmap[r, c, :]

        lbl_idx = int(lbl_memmap[r, c, 0])
        lbl_idx_batch[i] = lbl_idx

        # Build sub_state from superpixel
        sub_state = np.zeros(nstate)
        sub_state[idx_atmosphere] = x_atmosphere_batch[i, :]
        # Note: This requires access to fm.idx_surface which we'll handle in caller
        sub_state_batch[i, :] = sub_state

        valid_mask.append(i)

    # Trim to valid pixels
    if len(valid_mask) < n_pixels:
        meas_batch = meas_batch[valid_mask]
        loc_batch = loc_batch[valid_mask]
        obs_batch = obs_batch[valid_mask]
        if len(svf_batch) > 0:
            svf_batch = svf_batch[valid_mask]
        x_atmosphere_batch = x_atmosphere_batch[valid_mask]
        sub_state_batch = sub_state_batch[valid_mask]
        lbl_idx_batch = lbl_idx_batch[valid_mask]

    return (
        meas_batch,
        loc_batch,
        obs_batch,
        svf_batch,
        x_atmosphere_batch,
        sub_state_batch,
        lbl_idx_batch,
        valid_mask,
    )


def retrieve_winidx(config):
    wl_init, fwhm_init = load_wavelen(config.forward_model.instrument.wavelength_file)
    windows = config.implementation.inversion.windows

    winidx = np.array((), dtype=int)
    for lo, hi in windows:
        idx = np.where(np.logical_and(wl_init > lo, wl_init < hi))[0]
        winidx = np.concatenate((winidx, idx), axis=0)

    return winidx


def analytical_line(
    rdn_file: str,
    loc_file: str,
    obs_file: str,
    isofit_dir: str,
    isofit_config: str = None,
    segmentation_file: str = None,
    n_atm_neighbors: list = [20],
    n_cores: int = -1,
    num_iter: int = 1,
    smoothing_sigma: list = [2],
    output_rfl_file: str = None,
    output_unc_file: str = None,
    atm_file: str = None,
    skyview_factor_file: str = None,
    loglevel: str = "INFO",
    logfile: str = None,
    initializer: str = "algebraic",
    segmentation_size: int = 40,
    use_batched: bool = False,
    batch_size: int = 100,
) -> None:
    """
    TODO: Description
    """
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s ||| %(message)s",
        level=loglevel,
        filename=logfile,
        datefmt="%Y-%m-%d,%H:%M:%S",
    )

    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    # Config handling
    if isofit_config is None:
        file = glob(os.path.join(isofit_dir, "config", "") + "*_isofit.json")[0]
    else:
        file = isofit_config

    config = configs.create_new_config(file)
    config.forward_model.instrument.integrations = 1
    wl_init, fwhm_init = load_wavelen(config.forward_model.instrument.wavelength_file)

    # Set up input file paths
    subs_state_file = config.output.estimated_state_file
    subs_loc_file = config.input.loc_file

    # Rename files
    lbl_file = (
        segmentation_file
        if segmentation_file
        else (subs_state_file.replace("_subs_state", "_lbl"))
    )
    analytical_rfl_path = (
        output_rfl_file
        if output_rfl_file
        else (subs_state_file.replace("_subs_state", "_rfl"))
    )
    analytical_rfl_unc_path = (
        output_unc_file
        if output_unc_file
        else (subs_state_file.replace("_subs_state", "_uncert"))
    )

    # Files names for non-surface reflectance states
    analytical_non_rfl_surf_file = subs_state_file.replace(
        "_subs_state", "_surf_non_rfl"
    )
    analytical_non_rfl_surf_unc_file = subs_state_file.replace(
        "_subs_state", "_surf_non_rfl_uncert"
    )

    atm_file = (
        atm_file
        if atm_file
        else (subs_state_file.replace("_subs_state", "_atm_interp"))
    )

    # Get full statevector for image
    (
        full_statevector,
        full_idx_surface,
        full_idx_surf_rfl,
        _,
        full_idx_atmosphere,
        full_idx_instrument,
    ) = construct_full_state(config)

    # Perform the atmospheric interpolation
    if os.path.isfile(atm_file) is False:
        atm_interpolation(
            reference_state_file=subs_state_file,
            reference_locations_file=subs_loc_file,
            input_locations_file=loc_file,
            segmentation_file=lbl_file,
            output_atm_file=atm_file,
            atm_band_names=[full_statevector[i] for i in full_idx_atmosphere],
            nneighbors=n_atm_neighbors,
            gaussian_smoothing_sigma=smoothing_sigma,
            n_cores=n_cores,
        )

    # Get string representation of bad band list
    outside_ret_windows = np.zeros(len(full_idx_surf_rfl), dtype=int)
    outside_ret_windows[retrieve_winidx(config)] = 1

    # Get output shape
    rdn_ds = envi.open(envi_header(rdn_file))
    rdns = rdn_ds.shape
    rdn_meta = rdn_ds.metadata
    del rdn_ds

    # Construct surf rfl output
    output_metadata = {
        "data type": 4,
        "file type": "ENVI Standard",
        "byte order": 0,
        "no data value": -9999,
        "wavelength units": "Nanometers",
        "wavelength": wl_init,
        "fwhm": fwhm_init,
        "lines": rdn_meta["lines"],
        "samples": rdn_meta["samples"],
        "interleave": "bil",
    }
    if "map info" in rdn_meta:
        output_metadata["map info"] = (
            "{" + ", ".join(map(str, rdn_meta["map info"])) + "}"
        )

    output_metadata["band names"] = [
        full_statevector[i] for i in range(len(full_idx_surf_rfl))
    ]
    bbl = "{" + ",".join([f"{x}" for x in outside_ret_windows]) + "}"
    num_bands = len(full_idx_surf_rfl)
    engine_name = config.forward_model.atmosphere.engine_name
    isofit_version = config.implementation.isofit_version
    rfl_output = initialize_output(
        output_metadata,
        analytical_rfl_path,
        (rdns[0], num_bands, rdns[1]),
        bands=f"{num_bands}",
        bbl=bbl,
        description=(
            f"L2A Analytical per-pixel surface retrieval (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
        ),
    )

    # Construct surf rfl uncertainty output
    output_metadata["band names"] = [
        full_statevector[i] for i in range(len(full_idx_surf_rfl))
    ]
    unc_output = initialize_output(
        output_metadata,
        analytical_rfl_unc_path,
        (rdns[0], num_bands, rdns[1]),
        bands=f"{num_bands}",
        bbl=bbl,
        description=(
            f"L2A Analytical per-pixel surface retrieval uncertainty (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
        ),
    )

    # If there are more idx in surface than rfl, there are non_rfl surface states
    if len(full_idx_surface) > len(full_idx_surf_rfl):
        n_non_rfl_bands = len(full_idx_surface) - len(full_idx_surf_rfl)
        output_metadata["band names"] = [
            full_statevector[len(full_idx_surf_rfl) + i] for i in range(n_non_rfl_bands)
        ]
        non_rfl_output = initialize_output(
            output_metadata,
            analytical_non_rfl_surf_file,
            (rdns[0], n_non_rfl_bands, rdns[1]),
            bands=f"{n_non_rfl_bands}",
            description=(
                f"L2A Analytical per-pixel non_rfl surface retrieval  (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
            ),
        )

        non_rfl_unc_output = initialize_output(
            output_metadata,
            analytical_non_rfl_surf_unc_file,
            (rdns[0], n_non_rfl_bands, rdns[1]),
            bands=f"{n_non_rfl_bands}",
            description=(
                f"L2A Analytical per-pixel non_rfl surface retrieval uncertainty  (segmentation_size={segmentation_size}, engine={engine_name}, isofit_version={isofit_version})"
            ),
        )
    else:
        non_rfl_output = None
        non_rfl_unc_output = None

    # Ray initialization
    ray_dict = {
        "ignore_reinit_error": config.implementation.ray_ignore_reinit_error,
        "address": config.implementation.ip_head,
        "_temp_dir": config.implementation.ray_temp_dir,
        "include_dashboard": config.implementation.ray_include_dashboard,
        "_redis_password": config.implementation.redis_password,
        "num_cpus": n_cores,
    }
    ray.init(**ray_dict)
    n_workers = n_cores

    # Set up the memory-contiguous multi-state pixel map by sub
    index_pairs = np.empty((rdns[0] * rdns[1], 2), dtype=int)
    meshgrid = np.meshgrid(*(range(rdns[0]), range(rdns[1])))
    index_pairs[:, 0] = meshgrid[0].flatten(order="f")
    index_pairs[:, 1] = meshgrid[1].flatten(order="f")
    del meshgrid

    cache_atmosphere = None
    input_config = deepcopy(config)
    surface_index = index_spectra_by_surface(
        input_config, index_pairs, force_full_res=True
    )
    for i, (surface_class_str, class_idx_pairs) in enumerate(surface_index.items()):
        # Handle multisurface
        config = update_config_for_surface(deepcopy(input_config), surface_class_str)

        fm = ForwardModel(config, cache_atmosphere)
        fm.match_statevector(full_statevector)

        # Initialize workers
        wargs = [ray.put(obj) for obj in (config, fm)]
        wargs += [
            surface_class_str,
            class_idx_pairs,
            full_statevector,
            full_idx_surface,
            full_idx_surf_rfl,
            full_idx_atmosphere,
            rdn_file,
            loc_file,
            obs_file,
            atm_file,
            subs_state_file,
            lbl_file,
            rfl_output,
            unc_output,
            non_rfl_output,
            non_rfl_unc_output,
            num_iter,
            loglevel,
            logfile,
            initializer,
            skyview_factor_file,
        ]
        workers = ray.util.ActorPool([Worker.remote(*wargs) for _ in range(n_workers)])

        line_breaks = np.linspace(
            0,
            rdns[0],
            n_workers * config.implementation.task_inflation_factor,
            dtype=int,
        )

        line_breaks = [
            (line_breaks[n], line_breaks[n + 1]) for n in range(len(line_breaks) - 1)
        ]

        # run workers
        start_time = time.time()
        if use_batched:
            results = list(
                workers.map_unordered(
                    lambda a, b: a.run_chunks_batched.remote(b, batch_size=batch_size),
                    line_breaks,
                )
            )
        else:
            results = list(
                workers.map_unordered(lambda a, b: a.run_chunks.remote(b), line_breaks)
            )

        # Cache atmosphere
        if not i:
            cache_atmosphere = fm.atmosphere

        del fm

    total_time = time.time() - start_time

    logging.info(
        f"Analytical line inversions complete.  {round(total_time,2)}s total, "
        f"{round(rdns[0]*rdns[1]/total_time,4)} spectra/s, "
        f"{round(rdns[0]*rdns[1]/total_time/n_cores, 4)} spectra/s/core"
    )


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(
        self,
        config: Config,
        fm: ForwardModel,
        surface_class_str: str,
        class_idx_pairs: np.array,
        full_statevector: list,
        full_idx_surface: np.array,
        full_idx_surf_rfl: np.array,
        full_idx_atmosphere: np.array,
        rdn_file: str,
        loc_file: str,
        obs_file: str,
        atm_file: str,
        subs_state_file: str,
        lbl_file: str,
        rfl_output: str,
        unc_output: str,
        non_rfl_output: str,
        non_rfl_unc_output: str,
        num_iter: int,
        loglevel: str,
        logfile: str,
        initializer: str,
        skyview_factor_file: str,
    ):
        """
        Worker class to help run a subset of spectra.
        Args:
            fm: isofit forward_model
            loglevel: output logging level
            logfile: output logging file
        """
        logging.basicConfig(
            format="%(levelname)s:%(asctime)s ||| %(message)s",
            level=loglevel,
            filename=logfile,
            datefmt="%Y-%m-%d,%H:%M:%S",
        )

        # Persist config
        self.config = config

        # Persist forward model
        self.fm = fm

        # Persist surface class (or all)
        self.surface_class_str = surface_class_str
        self.class_idx_pairs = class_idx_pairs

        # Will fail if env.data isn't set up
        self.esd = load_esd()

        self.full_statevector = full_statevector
        self.full_idx_surface = full_idx_surface
        self.full_idx_surf_rfl = full_idx_surf_rfl
        self.full_idx_atmosphere = full_idx_atmosphere
        self.n_rfl_bands = len(full_idx_surf_rfl)
        self.n_non_rfl_bands = len(full_idx_surface) - len(full_idx_surf_rfl)

        self.winidx = retrieve_winidx(self.config)

        # input arrays
        self.rdn = envi.open(envi_header(rdn_file)).open_memmap(interleave="bip")
        self.loc = envi.open(envi_header(loc_file)).open_memmap(interleave="bip")
        self.obs = envi.open(envi_header(obs_file)).open_memmap(interleave="bip")
        self.rt_state = envi.open(envi_header(atm_file)).open_memmap(interleave="bip")
        self.subs_state = envi.open(envi_header(subs_state_file)).open_memmap(
            interleave="bip"
        )
        self.lbl = envi.open(envi_header(lbl_file)).open_memmap(interleave="bip")

        # Open skyview file for ALAlg, or create an array of 1s.
        if skyview_factor_file:
            self.svf = envi.open(envi_header(skyview_factor_file)).open_memmap(
                interleave="bip"
            )
        else:
            self.svf = []

        # Lines and samples
        self.n_lines = self.rdn.shape[0]
        self.n_samples = self.rdn.shape[1]

        # output paths
        self.rfl_outpath = rfl_output
        self.unc_outpath = unc_output
        self.non_rfl_outpath = non_rfl_output
        self.non_rfl_unc_outpath = non_rfl_unc_output

        self.completed_spectra = 0
        self.hash_table = OrderedDict()
        self.hash_size = config.implementation.max_hash_table_size

        # Can't see any reason to leave these as optional
        self.subs_state_file = subs_state_file
        self.lbl_file = lbl_file

        # If I only want to use some of the atm_interp bands
        # Empty if all
        self.atm_bands = []

        # How many iterations to use for invert_analytical
        self.num_iter = num_iter

        # Define coszen for geom creation
        self.coszen = fm.atmosphere.coszen

        if config.input.radiometry_correction_file is not None:
            self.radiance_correction, wl = load_spectrum(
                config.input.radiometry_correction_file
            )
        else:
            self.radiance_correction = None

        self.initializer = initializer

    def run_chunks_batched(
        self, line_breaks: tuple, fill_value: float = -9999.0, batch_size: int = 100
    ) -> None:
        """
        Batched version of run_chunks that processes multiple pixels at once.

        Args:
            line_breaks: (start_line, stop_line) tuple
            fill_value: Fill value for invalid pixels
            batch_size: Number of pixels to process in each batch
        """
        # Profiling timers
        profile_times = {
            "io_read": 0.0,
            "geometry_creation": 0.0,
            "invert_algebraic": 0.0,
            "invert_analytical": 0.0,
            "state_fill": 0.0,
            "io_write": 0.0,
            "total": 0.0,
        }
        chunk_start = time.time()

        # Unpack arguments
        start_line, stop_line = line_breaks

        # Set up outputs
        output_rfl = (
            envi.open(envi_header(self.rfl_outpath))
            .open_memmap(interleave="bip", writable=False)[start_line:stop_line, ...]
            .copy()
        )

        output_rfl_unc = (
            envi.open(envi_header(self.unc_outpath))
            .open_memmap(interleave="bip", writable=False)[start_line:stop_line, ...]
            .copy()
        )

        if self.non_rfl_unc_outpath:
            output_non_rfl = (
                envi.open(envi_header(self.non_rfl_outpath))
                .open_memmap(interleave="bip", writable=False)[
                    start_line:stop_line, ...
                ]
                .copy()
            )

            output_non_rfl_unc = (
                envi.open(envi_header(self.non_rfl_unc_outpath))
                .open_memmap(interleave="bip", writable=False)[
                    start_line:stop_line, ...
                ]
                .copy()
            )

        # Find intersection between index_pairs and class_idx_pairs
        index_pairs = self.class_idx_pairs[
            np.where(
                (self.class_idx_pairs[:, 0] >= start_line)
                & (self.class_idx_pairs[:, 0] < stop_line)
            )
        ]

        n_pixels = len(index_pairs)
        n_batches = (n_pixels + batch_size - 1) // batch_size

        # Process in batches
        for batch_idx in range(n_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = min((batch_idx + 1) * batch_size, n_pixels)
            batch_pairs = index_pairs[batch_start_idx:batch_end_idx]

            if len(batch_pairs) == 0:
                continue

            # Read batch of pixels
            t0 = time.time()
            (
                meas_batch,
                loc_batch,
                obs_batch,
                svf_batch,
                x_atmosphere_batch,
                sub_state_batch,
                lbl_idx_batch,
                valid_mask,
            ) = batch_read_pixels(
                batch_pairs,
                self.rdn,
                self.loc,
                self.obs,
                self.rt_state,
                self.svf,
                self.subs_state,
                self.lbl,
                self.fm.surface.analytical_iv_idx,
                self.fm.idx_atmosphere,
                self.fm.idx_instrument,
                self.fm.nstate,
                self.radiance_correction,
            )
            profile_times["io_read"] += time.time() - t0

            if len(valid_mask) == 0:
                continue

            # Complete sub_state construction
            for i, lbl_idx in enumerate(lbl_idx_batch):
                sub_state_batch[i, self.fm.idx_surface] = self.subs_state[
                    lbl_idx, 0, self.fm.surface.analytical_iv_idx
                ]
                sub_state_batch[i, self.fm.idx_instrument] = self.subs_state[
                    lbl_idx, 0, self.fm.idx_instrument
                ]
                sub_state_batch[i][np.isnan(sub_state_batch[i])] = self.fm.init[
                    np.isnan(sub_state_batch[i])
                ]

            # Create geometries
            t0 = time.time()
            geom_batch = batch_create_geometries(
                obs_batch,
                loc_batch,
                svf_batch,
                self.esd,
                self.coszen,
                self.config,
            )
            profile_times["geometry_creation"] += time.time() - t0

            # Initialize x0 batch
            x0_batch = np.zeros((len(valid_mask), self.fm.nstate))

            if self.initializer == "superpixel":
                x0_batch = sub_state_batch.copy()
                for i in range(len(valid_mask)):
                    x0_batch[i, self.fm.idx_atmosphere] = x_atmosphere_batch[i, :]

            elif self.initializer == "algebraic":
                t0 = time.time()
                for i, geom in enumerate(geom_batch):
                    x_surface, _, x_instrument = self.fm.unpack(self.fm.init.copy())
                    rfl_est, coeffs = invert_algebraic(
                        self.fm,
                        x_surface,
                        x_atmosphere_batch[i],
                        x_instrument,
                        meas_batch[i],
                        geom,
                    )

                    rfl_est = self.fm.surface.fit_params(rfl_est, geom)

                    x0_batch[i, :] = np.concatenate(
                        [
                            rfl_est,
                            x_atmosphere_batch[i],
                            x_instrument,
                        ]
                    )
                profile_times["invert_algebraic"] += time.time() - t0

            elif self.initializer == "simple":
                for i, geom in enumerate(geom_batch):
                    x0 = invert_simple(self.fm, meas_batch[i], geom)
                    x0[self.fm.idx_atmosphere] = x_atmosphere_batch[i]
                    x0_batch[i, :] = x0

            else:
                raise ValueError("No valid initializer given for AOE algorithm")

            # Set geom.x_surf_init for each geometry
            for i, geom in enumerate(geom_batch):
                geom.x_surf_init = x0_batch[i, self.fm.idx_surface]

            # Batch analytical inversion
            t0 = time.time()
            trajectories, uncertainties = batch_invert_analytical(
                self.fm,
                self.winidx,
                meas_batch,
                geom_batch,
                x0_batch,
                sub_state_batch,
                num_iter=self.num_iter,
            )
            profile_times["invert_analytical"] += time.time() - t0

            # Fill output arrays
            t0 = time.time()
            for i, valid_idx in enumerate(valid_mask):
                actual_idx = batch_start_idx + valid_idx
                r, c = batch_pairs[valid_idx][:2]

                state_est = trajectories[i, -1, :]
                unc = uncertainties[i, :]

                full_state_est = fill_statevector(
                    state_est,
                    self.fm.full_idx,
                    self.fm.full_miss,
                    self.full_statevector,
                )
                output_rfl[r - start_line, c, :] = full_state_est[
                    self.full_idx_surf_rfl
                ]

                full_unc_est = fill_statevector(
                    unc, self.fm.full_idx, self.fm.full_miss, self.full_statevector
                )
                output_rfl_unc[r - start_line, c, :] = full_unc_est[
                    self.full_idx_surf_rfl
                ]

                if self.non_rfl_outpath:
                    output_non_rfl[r - start_line, c, :] = full_state_est[
                        self.n_rfl_bands : self.n_rfl_bands + self.n_non_rfl_bands
                    ]
                    output_non_rfl_unc[r - start_line, c, :] = full_unc_est[
                        self.n_rfl_bands : self.n_rfl_bands + self.n_non_rfl_bands
                    ]
            profile_times["state_fill"] += time.time() - t0

        profile_times["total"] = time.time() - chunk_start

        logging.info(
            f"Analytical line chunk (BATCHED) {start_line}-{stop_line} ({n_pixels} pixels, {self.surface_class_str}): "
            f"total={profile_times['total']:.2f}s, "
            f"io_read={profile_times['io_read']:.2f}s ({profile_times['io_read']/profile_times['total']*100:.1f}%), "
            f"geom={profile_times['geometry_creation']:.2f}s ({profile_times['geometry_creation']/profile_times['total']*100:.1f}%), "
            f"alg_init={profile_times['invert_algebraic']:.2f}s ({profile_times['invert_algebraic']/profile_times['total']*100:.1f}%), "
            f"analytical={profile_times['invert_analytical']:.2f}s ({profile_times['invert_analytical']/profile_times['total']*100:.1f}%), "
            f"fill={profile_times['state_fill']:.2f}s ({profile_times['state_fill']/profile_times['total']*100:.1f}%), "
            f"rate={n_pixels/profile_times['total']:.1f} px/s"
        )

        t0 = time.time()
        # Output surface rfl
        write_bil_chunk(
            np.swapaxes(output_rfl, 1, 2),
            self.rfl_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )

        write_bil_chunk(
            np.swapaxes(output_rfl_unc, 1, 2),
            self.unc_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )

        if self.non_rfl_outpath:
            write_bil_chunk(
                np.swapaxes(output_non_rfl, 1, 2),
                self.non_rfl_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
            )
            write_bil_chunk(
                np.swapaxes(output_non_rfl_unc, 1, 2),
                self.non_rfl_unc_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
            )

        profile_times["io_write"] = time.time() - t0
        logging.info(
            f"Chunk {start_line}-{stop_line} write time: {profile_times['io_write']:.2f}s"
        )

    def run_chunks(self, line_breaks: tuple, fill_value: float = -9999.0) -> None:
        """
        TODO: Description
        """
        # Profiling timers
        profile_times = {
            "io_read": 0.0,
            "geometry_creation": 0.0,
            "invert_algebraic": 0.0,
            "invert_analytical": 0.0,
            "state_fill": 0.0,
            "io_write": 0.0,
            "total": 0.0,
        }
        chunk_start = time.time()

        # Unpack arguments
        start_line, stop_line = line_breaks

        # Set up outputs
        output_rfl = (
            envi.open(envi_header(self.rfl_outpath))
            .open_memmap(interleave="bip", writable=False)[start_line:stop_line, ...]
            .copy()
        )

        output_rfl_unc = (
            envi.open(envi_header(self.unc_outpath))
            .open_memmap(interleave="bip", writable=False)[start_line:stop_line, ...]
            .copy()
        )

        if self.non_rfl_unc_outpath:
            output_non_rfl = (
                envi.open(envi_header(self.non_rfl_outpath))
                .open_memmap(interleave="bip", writable=False)[
                    start_line:stop_line, ...
                ]
                .copy()
            )

            output_non_rfl_unc = (
                envi.open(envi_header(self.non_rfl_unc_outpath))
                .open_memmap(interleave="bip", writable=False)[
                    start_line:stop_line, ...
                ]
                .copy()
            )

        # Find intersection between index_pairs and class_idx_pairs
        index_pairs = self.class_idx_pairs[
            np.where(
                (self.class_idx_pairs[:, 0] >= start_line)
                & (self.class_idx_pairs[:, 0] < stop_line)
            )
        ]

        for r, c, *_ in index_pairs:
            t0 = time.time()
            meas = self.rdn[r, c, :]

            if self.radiance_correction is not None:
                meas = meas.copy() * self.radiance_correction

            if np.all(meas < 0):
                continue
            profile_times["io_read"] += time.time() - t0

            t0 = time.time()
            geom = Geometry(
                obs=self.obs[r, c, :],
                loc=self.loc[r, c, :],
                esd=self.esd,
                svf=self.svf[r, c] if len(self.svf) else 1,
                coszen=self.coszen,
                full_config=self.config,
            )
            profile_times["geometry_creation"] += time.time() - t0

            # "Atmospheric" state ALWAYS comes from all bands in the
            # atm_interpolated file
            x_atmosphere = self.rt_state[r, c, :]

            # TODO depricate this iv_idx. Abstract the indexing a bit more
            # s.t. we can smooth any statevector element by specifying idx
            # iv_idx here is a relic from a version that
            # achieved this by using atm_band_names in atm_interpolation
            # need to improve that implementation
            iv_idx = self.fm.surface.analytical_iv_idx

            # Populate the "background" superpixel
            lbl_idx = int(self.lbl[r, c, 0])
            sub_state = np.zeros(self.fm.nstate)
            sub_state[self.fm.idx_surface] = self.subs_state[lbl_idx, 0, iv_idx]
            sub_state[self.fm.idx_atmosphere] = x_atmosphere
            sub_state[self.fm.idx_instrument] = self.subs_state[
                lbl_idx, 0, self.fm.idx_instrument
            ]
            # Enforce non-NaN
            sub_state[np.isnan(sub_state)] = self.fm.init[np.isnan(sub_state)]

            # Build statevector to use for initialization.
            # Can be done three different ways.
            # SUPERPIXEL uses the superpixel value --> Fastests
            # ALGEBRAIC uses invert_algebraic for rfl,
            # and the superpixel for non_rfl surface elements
            # SIMPLE uses invert_simple for rfl and non_rfl surface elements
            if self.initializer == "superpixel":
                x0 = sub_state
                x0[self.fm.idx_atmosphere] = x_atmosphere

            elif self.initializer == "algebraic":
                t0 = time.time()
                x_surface, _, x_instrument = self.fm.unpack(self.fm.init.copy())
                rfl_est, coeffs = invert_algebraic(
                    self.fm,
                    x_surface,
                    x_atmosphere,
                    x_instrument,
                    meas,
                    geom,
                )

                rfl_est = self.fm.surface.fit_params(rfl_est, geom)

                x0 = np.concatenate(
                    [
                        rfl_est,
                        x_atmosphere,
                        x_instrument,
                    ]
                )
                profile_times["invert_algebraic"] += time.time() - t0

            elif self.initializer == "simple":
                x0 = invert_simple(self.fm, meas, geom)
                x0[self.fm.idx_atmosphere] = x_atmosphere

            else:
                raise ValueError("No valid initializer given for AOE algorithm")

            # NOTE: this line needs to be here to ensure geom.surf_cmp_init is populated
            geom.x_surf_init = x0[self.fm.idx_surface]

            t0 = time.time()
            states, unc = invert_analytical(
                self.fm,
                self.winidx,
                meas,
                geom,
                np.copy(x0),
                sub_state,
                num_iter=self.num_iter,
            )
            state_est = states[-1]
            profile_times["invert_analytical"] += time.time() - t0

            t0 = time.time()
            full_state_est = fill_statevector(
                state_est, self.fm.full_idx, self.fm.full_miss, self.full_statevector
            )
            output_rfl[r - start_line, c, :] = full_state_est[self.full_idx_surf_rfl]

            full_unc_est = fill_statevector(
                unc, self.fm.full_idx, self.fm.full_miss, self.full_statevector
            )
            output_rfl_unc[r - start_line, c, :] = full_unc_est[self.full_idx_surf_rfl]
            profile_times["state_fill"] += time.time() - t0

            full_state_est[len(self.full_idx_surf_rfl) : self.n_non_rfl_bands]
            # Save the non_rfl portion
            if self.non_rfl_outpath:
                output_non_rfl[r - start_line, c, :] = full_state_est[
                    self.n_rfl_bands : self.n_rfl_bands + self.n_non_rfl_bands
                ]
                output_non_rfl_unc[r - start_line, c, :] = full_unc_est[
                    self.n_rfl_bands : self.n_rfl_bands + self.n_non_rfl_bands
                ]

        profile_times["total"] = time.time() - chunk_start
        n_pixels = len(index_pairs)

        logging.info(
            f"Analytical line chunk {start_line}-{stop_line} ({n_pixels} pixels, {self.surface_class_str}): "
            f"total={profile_times['total']:.2f}s, "
            f"io_read={profile_times['io_read']:.2f}s ({profile_times['io_read']/profile_times['total']*100:.1f}%), "
            f"geom={profile_times['geometry_creation']:.2f}s ({profile_times['geometry_creation']/profile_times['total']*100:.1f}%), "
            f"alg_init={profile_times['invert_algebraic']:.2f}s ({profile_times['invert_algebraic']/profile_times['total']*100:.1f}%), "
            f"analytical={profile_times['invert_analytical']:.2f}s ({profile_times['invert_analytical']/profile_times['total']*100:.1f}%), "
            f"fill={profile_times['state_fill']:.2f}s ({profile_times['state_fill']/profile_times['total']*100:.1f}%), "
            f"rate={n_pixels/profile_times['total']:.1f} px/s"
        )

        t0 = time.time()
        # Output surface rfl
        write_bil_chunk(
            np.swapaxes(output_rfl, 1, 2),
            # output_rfl.T,
            self.rfl_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )

        # Save surface state uncertainty
        write_bil_chunk(
            np.swapaxes(output_rfl_unc, 1, 2),
            # output_rfl_unc.T,
            self.unc_outpath,
            start_line,
            (self.n_lines, self.n_rfl_bands, self.n_samples),
        )

        if self.non_rfl_outpath:
            write_bil_chunk(
                np.swapaxes(output_non_rfl, 1, 2),
                self.non_rfl_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
            )
            write_bil_chunk(
                np.swapaxes(output_non_rfl_unc, 1, 2),
                self.non_rfl_unc_outpath,
                start_line,
                (self.n_lines, self.n_non_rfl_bands, self.n_samples),
            )

        profile_times["io_write"] = time.time() - t0
        logging.info(
            f"Chunk {start_line}-{stop_line} write time: {profile_times['io_write']:.2f}s"
        )


@click.command(name="analytical_line")
@click.argument("rdn_file")
@click.argument("loc_file")
@click.argument("obs_file")
@click.argument("isofit_dir")
@click.option("--isofit_config", type=str, default=None)
@click.option("--segmentation_file", help="TODO", type=str, default=None)
@click.option("--n_atm_neighbors", help="TODO", type=int, default=20)
@click.option("--n_cores", help="TODO", type=int, default=-1)
@click.option("--smoothing_sigma", help="TODO", type=int, default=2)
@click.option("--output_rfl_file", help="TODO", type=str, default=None)
@click.option("--output_unc_file", help="TODO", type=str, default=None)
@click.option("--skyview_factor_file", help="TODO", type=str, default=None)
@click.option("--atm_file", help="TODO", type=str, default=None)
@click.option("--loglevel", help="TODO", type=str, default="INFO")
@click.option("--logfile", help="TODO", type=str, default=None)
@click.option(
    "--use_batched",
    help="Use batched vectorized processing",
    is_flag=True,
    default=False,
)
@click.option(
    "--batch_size", help="Batch size for vectorized processing", type=int, default=100
)
def cli(**kwargs):
    """Execute the analytical line algorithm"""

    click.echo("Running analytical line")

    analytical_line(**kwargs)

    click.echo("Done")


if __name__ == "__main__":
    raise NotImplementedError(
        "analytical_line.py can no longer be called this way.  Run as:\n isofit analytical_line [ARGS]"
    )
