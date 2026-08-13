# Lazy re-exports (PEP 562): urcirobot pulls in torch via humanoidverse.envs,
# but subpackages such as skill_scheduler must stay importable without torch.
# Existing call sites (`from humanoidverse.deploy import URCIRobot`, and
# `from humanoidverse.deploy import *` via __all__) behave exactly as before.

__all__ = ["URCIRobot", "ObsCfg", "URCIPolicyObs", "CfgType"]


def __getattr__(name):
    if name in __all__:
        from .urcirobot import URCIRobot, ObsCfg, URCIPolicyObs, CfgType
        return {
            "URCIRobot": URCIRobot,
            "ObsCfg": ObsCfg,
            "URCIPolicyObs": URCIPolicyObs,
            "CfgType": CfgType,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
