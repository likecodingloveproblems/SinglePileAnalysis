from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from openseespy import opensees as ops

from enums import OpsElements
from materials import (
    SoilProfile,
    PileFrictionMaterial,
    PileTipMaterial,
    PileStructureMaterial,
)
from tag_generator import TaggedObject


@dataclass
class Node(TaggedObject):
    depth: float

    def __post_init__(self):
        """Generate the object on the ops side"""
        ops.node(self.tag, self.depth)


@dataclass
class SoilElement(TaggedObject):
    fixed_node: Node
    pile_joined_node: Node
    material: PileFrictionMaterial | PileTipMaterial
    ops_element_type: OpsElements = OpsElements.ZeroLength.value

    def __post_init__(self):
        """
        First build nodes one that is fixed, another that be attached to pile structure
        Then build the material at the specified depth
        Then create zerolength element in ops
        :return:
        :rtype:
        """
        ops.element(
            self.tag,
            self.ops_element_type,
            self.fixed_node.tag,
            self.pile_joined_node.tag,
            "-mat",
            self.material.tag,
            "-dir",
            1,
        )


@dataclass
class PileElement(TaggedObject):
    first_node: Node
    second_node: Node
    pile_radius: float
    material: PileStructureMaterial
    ops_element_type: OpsElements = OpsElements.Truss.value

    @property
    def __area(self):
        return np.pi * (self.pile_radius**2)

    def __post_init__(self):
        """
        first create nodes
        then create the material
        then create the element of pile
        then create springs to model soil behavior
        then attach spring to pile by making equal degree of freedom
        """
        ops.element(
            self.ops_element_type,
            self.tag,
            self.first_node.tag,
            self.second_node.tag,
            self.__area,
            self.material,
        )


@dataclass
class CalibrationParams:
    Rfb: float
    Sbu: float
    alpha21: float
    Rfs: float


@dataclass
class Pile:
    pile_length: float
    pile_radius: float
    soil_profile: SoilProfile
    elasticity_modulus: float
    calibration_params: CalibrationParams
    load: float
    pile_structure_material: PileStructureMaterial = field(init=False)
    number_of_node: int = 100

    def __post_init__(self):
        self.pile_structure_material = PileStructureMaterial(
            elasticity_module=self.elasticity_modulus
        )
        self.__initialization()
        self.__mesh()
        self.__generate_elements()

    @staticmethod
    def __initialization():
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)

    @property
    def __pile_head_node(self):
        return list(filter(lambda node: node.depth == 0, self.pile_nodes))[0]

    def __mesh(self):
        depths = np.linspace(0, self.pile_length, self.number_of_node)
        self.pile_nodes = [Node(depth) for depth in depths]
        self.fixed_nodes = [Node(depth) for depth in depths]
        # fix them
        for node in self.fixed_nodes:
            ops.fix(node.tag, 1)

    def __is_tip_node(self, node: Node) -> bool:
        return node.depth == self.pile_length

    def __generate_elements(self):
        self.__generate_pile_structura_elements()
        self.__generate_soil_elements()
        self.__apply_load_at_pile_head()
        self.__analyze()

    def __generate_pile_structura_elements(self):
        """First create pile structure elements then assign soil springs to each node of pile structure"""
        self.pile_elements = list()
        for first_node, second_node in zip(self.pile_nodes, self.pile_nodes[1:]):
            self.pile_elements.append(
                PileElement(
                    first_node=first_node,
                    second_node=second_node,
                    pile_radius=self.pile_radius,
                    material=self.pile_structure_material,
                )
            )

    @property
    def __pile_element_length(self):
        return self.pile_length / self.number_of_node

    def __generate_soil_elements(self):
        """First generate spring materials
        then generate elements"""
        self.materials = list()
        self.soil_elements = list()
        for pile_node, fixed_node in zip(self.pile_nodes, self.fixed_nodes):
            if self.__is_tip_node(pile_node):
                soil_material = PileTipMaterial(
                    soil_profile=self.soil_profile,
                    pile_radius=self.pile_radius,
                    Rfb=self.calibration_params.Rfb,
                    Sbu=self.calibration_params.Sbu,
                    alpha21=self.calibration_params.alpha21,
                )
            else:
                soil_material = PileFrictionMaterial(
                    soil_profile=self.soil_profile,
                    pile_radius=self.pile_radius,
                    depth=pile_node.depth,
                    pile_element_length=self.__pile_element_length,
                    Rfs=self.calibration_params.Rfs,
                )
            self.materials.append(soil_material)
            self.soil_elements.append(
                SoilElement(
                    fixed_node=fixed_node,
                    pile_joined_node=pile_node,
                    material=soil_material,
                )
            )

    def __apply_load_at_pile_head(self):
        ops.load(self.__pile_head_node.tag, self.load)

    @staticmethod
    def __analyze():
        ops.system("BandSPD")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        ops.analyze(1)

    def __monitor(self):
        """Track displacement at pile head and force at pile head"""
        self.pile_head_displacement = ops.nodeDisp(self.__pile_head_node.tag, 1)
