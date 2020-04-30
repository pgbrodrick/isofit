
from typing import Dict, List, Type
from isofit.configs.base_config import BaseConfigSection
import logging
import numpy as np


class StateVectorElementConfig(BaseConfigSection):
    """
    State vector element configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._bounds_type = list()
        self.bounds = [np.nan, np.nan]

        self._scale_type = float
        self.scale = np.nan

        self._prior_mean_type = float
        self.prior_mean = np.nan

        self._prior_sigma_type = float
        self.prior_sigma = np.nan

        self._init_type = float
        self.init = np.nan

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()

        return errors

class StateVectorConfig(BaseConfigSection):
    """
    State vector configuration.
    """

    def __init__(self, sub_configdic: dict = None):
        self._H2OSTR_type = StateVectorElementConfig
        self.H2OSTR: StateVectorElementConfig = None

        self._AOT550_type = StateVectorElementConfig
        self.AOT550: StateVectorElementConfig = None

        self._AERFRAC_1_type = StateVectorElementConfig
        self.AERFRAC_1: StateVectorElementConfig = None

        self._AERFRAC_2_type = StateVectorElementConfig
        self.AERFRAC_2: StateVectorElementConfig = None

        self._AERFRAC_3_type = StateVectorElementConfig
        self.AERFRAC_3: StateVectorElementConfig = None

        self._GROW_FWHM_type = StateVectorElementConfig
        self.GROW_FWHM: StateVectorElementConfig = None

        self._WL_SHIFT_type = StateVectorElementConfig
        self.WL_SHIFT: StateVectorElementConfig = None

        self._WL_SPACE_type = StateVectorElementConfig
        self.WL_SPACE: StateVectorElementConfig = None

        assert(len(self.get_all_elements()) == len(self._get_nontype_attributes()))

        self._set_statevector_config_options(sub_configdic)

    def _check_config_validity(self):
        errors = list()

        return errors

    def _set_statevector_config_options(self, configdic):
        #TODO: update using methods below
        if configdic is not None:
            for key in configdic:
                sv = StateVectorElementConfig(configdic[key])
                setattr(self, key, sv)

    def get_all_elements(self):
        return [self.H2OSTR, self.AOT550, self.AERFRAC_1, self.AERFRAC_2, self.AERFRAC_3, self.GROW_FWHM, self.WL_SHIFT,
                self.WL_SPACE]

    def get_elements(self):
        elements = [x for x in self.get_all_elements() if x is not None]
        element_names = self._get_nontype_attributes()

        order = np.argsort(element_names)
        elements = [elements[idx] for idx in order]

        return elements, element_names

    def get_element_names(self):
        elements, element_names = self.get_elements()
        return element_names

    def get_single_element_by_name(self, name):
        elements, element_names = self.get_elements()
        return elements[element_names.index(name)]


    def get_all_bounds(self):
        bounds = []
        for element, name in zip(*self.get_elements()):
            bounds.append(element.bounds)
        return bounds

    def get_all_scales(self):
        scales = []
        for element, name in zip(*self.get_elements()):
                scales.append(element.scale)
        return scales

    def get_all_inits(self):
        inits = []
        for element, name in zip(*self.get_elements()):
            inits.append(element.init)
        return inits

    def get_all_prior_means(self):
        prior_means = []
        for element, name in zip(*self.get_elements()):
            prior_means.append(element.prior_mean)
        return prior_means

    def get_all_prior_sigmas(self):
        prior_sigmas = []
        for element, name in zip(*self.get_elements()):
            prior_sigmas.append(element.prior_sigma)
        return prior_sigmas



