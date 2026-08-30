"""World-model implementations loaded on demand to keep inference workers isolated."""

from importlib import import_module

_EXPORTS = {
    "DreamerV3WorldModel": (".dreamerv3_world_model", "DreamerV3WorldModel"),
    "build_world_model": (".factory", "build_world_model"),
    "GRUWorldModel": (".gru_world_model", "GRUWorldModel"),
    "ActionConditionedJEPAWorldModel": (
        ".jepa_world_model",
        "ActionConditionedJEPAWorldModel",
    ),
    "OccFlowWorldModel": (".occflow_world_model", "OccFlowWorldModel"),
    "REQUIRED_OUTPUTS": (".protocol", "REQUIRED_OUTPUTS"),
    "validate_world_model_output": (".protocol", "validate_world_model_output"),
    "WorldModelLoss": (".losses", "WorldModelLoss"),
    "TransformerWorldModel": (".transformer_world_model", "TransformerWorldModel"),
    "mc_dropout_predict": (".uncertainty", "mc_dropout_predict"),
    "CandidateWorldModelRuntime": (".runtime", "CandidateWorldModelRuntime"),
    "ContinuousWorldModelPolicy": (
        ".continuous_protocol",
        "ContinuousWorldModelPolicy",
    ),
    "TDMPC2ContinuousPolicy": (".tdmpc2_continuous", "TDMPC2ContinuousPolicy"),
    "TDMPC2CandidateAssistant": (
        ".tdmpc2_candidate",
        "TDMPC2CandidateAssistant",
    ),
    "VisualTDMPC2CandidateAssistant": (
        ".tdmpc2_candidate",
        "VisualTDMPC2CandidateAssistant",
    ),
    "candidate_actions_body_flu": (
        ".tdmpc2_candidate",
        "candidate_actions_body_flu",
    ),
    "TDMPC2VisualNetwork": (".tdmpc2_visual", "TDMPC2VisualNetwork"),
    "TDMPC2VisualPolicy": (".tdmpc2_visual", "TDMPC2VisualPolicy"),
    "VisualTDMPC2Trainer": (".tdmpc2_visual_training", "VisualTDMPC2Trainer"),
    "V3CandidateWorldModelRuntime": (".v3_candidate_runtime", "V3CandidateWorldModelRuntime"),
    "OfficialVJEPA21Encoder": (".vjepa2_official", "OfficialVJEPA21Encoder"),
    "UAVActionConditionedJEPAPredictor": (".vjepa2_official", "UAVActionConditionedJEPAPredictor"),
    "enable_lora_last_blocks": (".vjepa2_official", "enable_lora_last_blocks"),
    "VJEPA21CandidateAssistant": (".vjepa2_official", "VJEPA21CandidateAssistant"),
    "vjepa_uav_training_loss": (".vjepa2_official", "vjepa_uav_training_loss"),
    "ARRFlyActorCritic": (".arr_fly_policy", "ARRFlyActorCritic"),
    "ARRFlyPPOPolicy": (".arr_fly_policy", "ARRFlyPPOPolicy"),
    "asymmetric_ppo_loss": (".arr_fly_policy", "asymmetric_ppo_loss"),
    "DreamerRSSMV3Network": (".dreamer_rssm_v3", "DreamerRSSMV3Network"),
    "DreamerRSSMV3Trainer": (".dreamer_rssm_v3", "DreamerRSSMV3Trainer"),
    "DreamerRSSMV3CandidateAssistant": (".dreamer_rssm_v3", "DreamerRSSMV3CandidateAssistant"),
    "WorldModelBase": (".base", "WorldModelBase"),
    "JEPAWorldModelAdapter": (".jepa_adapter", "JEPAWorldModelAdapter"),
    "VJEPAWorldModelAdapter": (".vjepa_wam", "VJEPAWorldModelAdapter"),
    "VJEPAWAMLossWeights": (".vjepa_wam", "VJEPAWAMLossWeights"),
    "vjepa_wam_multistep_loss": (".vjepa_wam", "vjepa_wam_multistep_loss"),
    "save_vjepa_wam_checkpoint": (".vjepa_wam", "save_vjepa_wam_checkpoint"),
    "load_vjepa_wam_checkpoint": (".vjepa_wam", "load_vjepa_wam_checkpoint"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
