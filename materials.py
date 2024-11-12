from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openseespy import opensees as ops

from enums import OpsMaterials
from tag_generator import TaggedObject


# we calculate frictional stiffness with closed form solutions and soil parameters
# first we need a function to calculate G (Shear modulus in depth), poisson ratio and
# Ultimate shear stress with depth
# Gs(z) -> shear modulus with respect to depth
# vs(z) -> poisson ratio with respect to depth
# Tau_f(z) -> Ultimate friction stress with respect to depth


@dataclass
class SoilLayer:
    up_depth: float
    bottom_depth: float
    up_shear_modulus: float
    bottom_shear_modulus: float
    up_poisson_ratio: float
    bottom_poisson_ratio: float
    up_tau_f: float
    bottom_tau_f: float

    @property
    def length(self):
        return self.up_depth - self.bottom_depth

    @property
    def avg_poisson_ratio(self):
        return (self.up_poisson_ratio + self.bottom_poisson_ratio) / 2

    @property
    def max_shear_modulus(self) -> float:
        return max(self.up_shear_modulus, self.bottom_shear_modulus)

    @property
    def avg_shear_modulus(self):
        return (self.up_shear_modulus + self.bottom_shear_modulus) / 2

    def shear_modulus(self, depth: float) -> float:
        return self.up_shear_modulus + (
            self.bottom_shear_modulus - self.up_shear_modulus
        ) / (self.bottom_depth - self.up_depth) * (depth - self.up_depth)

    def poisson_ratio(self, depth: float) -> float:
        return self.up_poisson_ratio + (
            self.bottom_poisson_ratio - self.up_poisson_ratio
        ) / (self.bottom_depth - self.up_depth) * (depth - self.up_depth)

    def tau_f(self, depth: float) -> float:
        return self.up_tau_f + (self.bottom_tau_f - self.up_tau_f) / (
            self.bottom_depth - self.up_depth
        ) * (depth - self.up_depth)


@dataclass
class SoilProfile:
    layers: list[SoilLayer]
    soil_layers_count: int

    @property
    def max_shear_modulus(self):
        return max(layer.max_shear_modulus for layer in self.layers)

    @property
    def pile_length(self):
        """It's assumed that SoilProfile contains soil layers that are at the length of pile"""
        return sum(layer.length for layer in self.layers)

    @property
    def avg_poisson_ratio(self):
        """How to calculate avg poisson ratio
        Simple avg: sum(layer.poisson_ratio) / count of layers
        Weighted avg: sum(layer.poisson_ratio * layer.length) / sum(layer.length)
        """
        return (
            sum(layer.avg_poisson_ratio * layer.length for layer in self.layers)
            / self.pile_length
        )

    def __find_layer(self, depth: float) -> SoilLayer:
        for layer in self.layers:
            if layer.up_depth <= depth <= layer.bottom_depth:
                return layer
        raise Exception(f"Not found layer for depth: {depth}!")

    def shear_modulus(self, depth: float) -> float:
        layer = self.__find_layer(depth)
        return layer.shear_modulus(depth)

    def poisson_ratio(self, depth: float) -> float:
        layer = self.__find_layer(depth)
        return layer.poisson_ratio(depth)

    def tau_f(self, depth: float) -> float:
        layer = self.__find_layer(depth)
        return layer.tau_f(depth)

    @property
    def tip_shear_modulus(self):
        return self.shear_modulus(self.pile_length)

    @property
    def tip_poisson_ratio(self):
        return self.poisson_ratio(self.pile_length)


@dataclass
class PileFrictionMaterial(TaggedObject):
    soil_profile: SoilProfile
    pile_radius: float
    depth: float
    pile_element_length: float
    Rfs: float
    ops_material_type: OpsMaterials = OpsMaterials.ElasticMultiLinear.value

    @property
    def pile_diameter(self):
        return 2 * np.pi * self.pile_radius

    @property
    def pile_element_surface_area(self):
        return self.pile_element_length * self.pile_diameter

    @property
    def rou_m(self) -> float:
        """Coefficient of enhanced ratio
        It's not related to depth
        C. Y. Lee, “Settlement of pile groups—practical approach,” J. Geotech. Eng., Vol. 119, No. 9, pp. 1449–1461, 1993.
        """
        return sum(
            layer.avg_shear_modulus
            * layer.length
            / (self.soil_profile.max_shear_modulus * self.soil_profile.pile_length)
            for layer in self.soil_profile.layers
        )

    @property
    def rm(self):
        return (
            2.5
            * self.soil_profile.pile_length
            * self.rou_m
            * (1 - self.soil_profile.avg_poisson_ratio)
        )

    def Ke_friction(self) -> float:
        """Initial stiffness of pile side friction in (kN/m3)"""
        return (
            self.soil_profile.shear_modulus(self.depth)
            / (self.pile_radius * np.log(self.rm / self.pile_radius))
            * self.pile_element_surface_area
        )

    def tau_ult(self):
        tau_su = self.soil_profile.tau_f(self.depth)
        return (tau_su / self.Rfs) * self.pile_element_surface_area

    def b(self):
        return 1 / self.tau_ult()

    def a(self):
        return 1 / self.Ke_friction()

    @staticmethod
    def __get_sample_displacements():
        return np.geomspace(1e-12, 0.1, 50)

    def __get_forces(self, strains):
        return strains / (self.a() + self.b() * strains)

    def __post_init__(self):
        strains = self.__get_sample_displacements()
        stresses = self.__get_forces(strains)
        ops.uniaxialMaterial(
            self.tag, self.ops_material_type, "-strain", *strains, "-stress", *stresses
        )


@dataclass
class PileTipMaterial(TaggedObject):
    soil_profile: SoilProfile
    pile_radius: float
    Rfb: float  # Corrective factor to apply, construction effects, soil type and ...
    Sbu: float  # final settlement to transfer to plastic phase
    alpha21: float  # ratio of second stiffness to initial stiffness
    ops_material_type = OpsMaterials.ElasticMultiLinear.value

    @property
    def depth(self):
        return self.soil_profile.pile_length

    @property
    def pile_tip_area(self):
        return np.pi * self.pile_radius**2

    @property
    def k1b(self) -> float:
        """Initial stiffness of pile tip without applying corrective factor
        Output unit: kN/m"""
        return (
            4
            * self.soil_profile.tip_shear_modulus
            / (np.pi * self.pile_radius * (1 - self.soil_profile.tip_poisson_ratio))
        ) * self.pile_tip_area

    @property
    def k1(self):
        """Rfb is a corrective factor
        so k1 applied effects of construction methods, soil type and ..."""
        return self.k1b / self.Rfb

    @property
    def k2(self) -> float:
        return self.k1 * self.alpha21

    def __get_strains(self):
        return np.array([0.0, self.Sbu, 0.1])

    def __get_stresses(self):
        return np.array(
            [0.0, self.Sbu * self.k1, self.Sbu * self.k1 + (0.1 - self.Sbu) * self.k2]
        )

    def __post_init__(self):
        strains = self.__get_strains()
        stresses = self.__get_stresses()
        ops.uniaxialMaterial(
            self.tag, self.ops_material_type, "-strain", *strains, "-stress", *stresses
        )


@dataclass
class PileStructureMaterial(TaggedObject):
    elasticity_module: float

    def __post_init__(self):
        ops.uniaxialMaterial(
            self.tag, OpsMaterials.Elastic.value, self.elasticity_module
        )
