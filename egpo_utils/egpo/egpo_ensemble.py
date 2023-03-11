from egpo_utils.egpo.egpo import EGPOTrainer
from egpo_utils.egpo.egpo_ensemble_policy import EGPOEnsemblePolicy

def get_policy_class(config):
    return EGPOEnsemblePolicy

EGPOEnsembleTrainer = EGPOTrainer.with_updates(
	name="EGPOEnsembleTrainer",
	default_policy=EGPOEnsemblePolicy,
	get_policy_class=get_policy_class,
)
