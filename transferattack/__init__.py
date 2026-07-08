import importlib


attack_zoo = {
    "agentic_schedule": (
        ".agentic_schedule",
        "AgenticScheduleAttack",
    ),
    "agent_quality_mdcsops": (
        ".input_transformation.agent_quality_mdcsops",
        "AgentQualityMDCSOPS",
    ),
    "mdcsops": (".input_transformation.mdcsops", "MDCSOPS"),
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise ValueError(f"Unsupported attack algorithm: {attack_name}")
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    return getattr(module, class_name)


__version__ = "1.0.0"
