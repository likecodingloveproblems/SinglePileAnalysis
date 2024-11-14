from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from openseespy import opensees as ops
from pydantic import BaseModel

from enums import OpsElements
from materials import (
    SoilProfile,
    PileFrictionMaterial,
    PileTipMaterial,
    PileStructureMaterial,
)
from tag_generator import TaggedObject


class Node(TaggedObject):
    depth: float

    def model_post_init(self, __context: Any) -> None:
        """Generate the object on the ops side"""
        ops.node(self.tag, self.depth)


class SoilElement(TaggedObject):
    fixed_node: Node
    pile_joined_node: Node
    material: PileFrictionMaterial | PileTipMaterial
    ops_element_type: OpsElements = OpsElements.ZeroLength.value

    def model_post_init(self, __context: Any) -> None:
        """
        First build nodes one that is fixed, another that be attached to pile structure
        Then build the material at the specified depth
        Then create zerolength element in ops
        :return:
        :rtype:
        """
        ops.element(
            self.ops_element_type,
            self.tag,
            self.fixed_node.tag,
            self.pile_joined_node.tag,
            "-mat",
            self.material.tag,
            "-dir",
            1,
        )


class PileElement(TaggedObject):
    first_node: Node
    second_node: Node
    pile_radius: float
    material: PileStructureMaterial
    area: float = None
    ops_element_type: OpsElements = OpsElements.Truss.value

    @property
    def __area(self):
        if self.area:
            return self.area
        return np.pi * (self.pile_radius**2)

    def model_post_init(self, __context: Any) -> None:
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
            self.material.tag,
        )


class CalibrationParams(BaseModel):
    Rfb: float
    Sbu: float
    alpha21: float
    Rfs: float
    bounds: ClassVar[list[tuple[float]]] = [
        (0.8, 1.0),
        (1e-3, 9e-3),
        (0.0, 1.0),
        (0.8, 1.0),
    ]

    @staticmethod
    def from_array(args):
        return CalibrationParams(Rfb=args[0], Sbu=args[1], alpha21=args[2], Rfs=args[3])

    @staticmethod
    def from_default():
        return CalibrationParams(Rfb=1.0, Sbu=5e-3, alpha21=0.01, Rfs=1.0)


class Pile(BaseModel):
    pile_length: float
    pile_radius: float
    soil_profile: SoilProfile
    elasticity_modulus: float
    calibration_params: CalibrationParams
    load: float
    area: float | None = None
    pile_structure_material: PileStructureMaterial = None
    number_of_node: int = 50
    number_of_steps: int = 300
    pile_nodes: list[Node] = []
    fixed_nodes: list[Node] = []
    materials: list[PileFrictionMaterial | PileTipMaterial] = []
    soil_elements: list[SoilElement] = []

    def model_post_init(self, __context: Any) -> None:
        self.__initialization()
        self.pile_structure_material = PileStructureMaterial(
            elasticity_module=self.elasticity_modulus
        )
        self.__mesh()
        self.__generate_elements()
        self.apply_load_at_pile_head()

    @staticmethod
    def __initialization():
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)

    @property
    def __pile_head_node(self):
        return list(filter(lambda node: node.depth == 0, self.pile_nodes))[0]

    def __mesh(self):
        depths = np.linspace(0, self.pile_length, self.number_of_node)
        self.pile_nodes = [Node(depth=depth) for depth in depths]
        self.fixed_nodes = [Node(depth=depth) for depth in depths]
        # fix them
        for node in self.fixed_nodes:
            ops.fix(node.tag, 1)

    def __is_tip_node(self, node: Node) -> bool:
        return node.depth == self.pile_length

    def __generate_elements(self):
        self.__generate_pile_structura_elements()
        self.__generate_soil_elements()

    def __generate_pile_structura_elements(self):
        """First create pile structure elements then assign soil springs to each node of pile structure"""
        for first_node, second_node in zip(self.pile_nodes, self.pile_nodes[1:]):
            PileElement(
                first_node=first_node,
                second_node=second_node,
                pile_radius=self.pile_radius,
                material=self.pile_structure_material,
                area=self.area,
            )

    @property
    def __pile_element_length(self):
        return self.pile_length / self.number_of_node

    def __generate_soil_elements(self):
        """First generate spring materials
        then generate elements"""
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

    def apply_load_at_pile_head(self):
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(self.__pile_head_node.tag, self.load)

    def analyze(self):
        ops.system("BandSPD")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0 / self.number_of_steps)
        ops.algorithm("Newton")
        ops.analysis("Static")
        head_displacement = []
        head_force = list()
        for step in range(self.number_of_steps):
            ops.analyze(1)
            head_force.append(ops.getLoadFactor(1) * self.load)
            head_displacement.append(ops.nodeDisp(self.__pile_head_node.tag, 1))
        return [
            head_displacement,
            head_force,
        ]
