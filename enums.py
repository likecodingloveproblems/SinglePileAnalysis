import enum


class OpsMaterials(enum.Enum):
    Elastic = "Elastic"
    ElasticMultiLinear = "ElasticMultiLinear"


class OpsElements(enum.Enum):
    Truss = "Truss"
    ZeroLength = "zeroLength"
