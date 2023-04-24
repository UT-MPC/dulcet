from plugins.evaluator.DiscoveryRateLogger import DiscoveryRateLogger
from plugins.evaluator.AdaptationErrorLogger import AdaptationErrorLogger
from plugins.evaluator.ContinuousDiscoveryRateLogger import ContinuousDiscoveryRateLogger
from plugins.evaluator.RollingWindowLogger import RollingWindowLogger
from plugins.evaluator.NdiffDistributionLogger import NdiffDistributionLogger
from plugins.evaluator.NdaLogger import NdaLogger
from plugins.evaluator.NdiffWindowLogger import NdiffWindowLogger
from plugins.evaluator.NodeSchedulePlotter import NodeSchedulePlotter
from plugins.evaluator.EpochEnergyLogger import EpochEnergyLogger

def get_evaluators(evaluator_dict, scene=None):
    evaluators = []

    # parse out the dict of evaluators into intantiated objects
    for evaluator_class, kwargs in evaluator_dict.items():
        evaluator = globals()[evaluator_class]

        # instantiate the evalautor, give it a reference to the scene, and append it to the list of evaluators
        instance = evaluator(**kwargs)
        instance.set_scene(scene)
        
        evaluators.append(instance)

    return evaluators