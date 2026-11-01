from __future__ import annotations

import torch
import torch.multiprocessing as mp
from typing import TYPE_CHECKING, List, Dict

from configuration import config
from src.phenotype.neural_network.evaluator.evaluator import evaluate
from src.phenotype.neural_network.feature_multiplication import get_model_of_target_size
from src.phenotype.neural_network.neural_network import Network

import src.main.singleton as singleton

if TYPE_CHECKING:
    from src.genotype.cdn.genomes.blueprint_genome import BlueprintGenome


def _is_cuda_oom(err: BaseException) -> bool:
    oom_types = tuple(t for t in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None))
                      if t is not None)  # torch.OutOfMemoryError only exists from PyTorch 2.5
    return isinstance(err, oom_types) or "out of memory" in str(err).lower()


def evaluate_blueprints(blueprint_q: mp.Queue,
                        input_size: List[int]) -> List[BlueprintGenome]:
    """
    Consumes blueprints off the blueprints queue, evaluates them and adds them back to the queue if all of their
    evaluations have not been completed for the current generation. If all their evaluations have been completed, add
    them to the completed_blueprints list.

    :param blueprint_q: A thread safe queue of blueprints
    :param input_size: The shape of the input to each network
    :param num_epochs: the number of epochs to train each model for
    :return: A list of evaluated blueprints
    """
    completed_blueprints: List[BlueprintGenome] = []
    print(f'Process {mp.current_process().name} - epochs: {config.epochs_in_evolution}')
    while blueprint_q.qsize() != 0:
        blueprint = blueprint_q.get()
        blueprint = evaluate_blueprint(blueprint, input_size)
        if blueprint.n_evaluations == config.n_evals_per_bp:
            completed_blueprints.append(blueprint)
        else:
            blueprint_q.put(blueprint)

    return completed_blueprints


def evaluate_blueprint(blueprint: BlueprintGenome, input_size: List[int],
                       feature_multiplier: float = 1) -> BlueprintGenome:
    """
    Parses the blueprint into its phenotype NN
    Handles the assignment of the single/multi obj finesses to the blueprint in parallel
    """

    allow_ignores = blueprint.n_evaluations == 0
    try:
        model: Network = Network(blueprint, input_size, feature_multiplier=feature_multiplier,
                                 allow_module_map_ignores=allow_ignores)
    except RuntimeError as exc:
        print(f"dropped blueprint {blueprint.id} before training: {exc}")
        blueprint.update_best_sample_map({}, 0)
        blueprint.report_fitness([0, 0])
        print("Blueprint - {:^5} - accuracy: {:05.2f}% (proc {}) epochs: {}"
              .format(blueprint.id, 0, mp.current_process().name, config.epochs_in_evolution))
        return blueprint
    # Built on the CPU so that an oversized net is dropped before it touches the GPU.
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sample_map = model.sample_map
    if config.target_network_size != -1:
        model = get_model_of_target_size(blueprint, sample_map, model_size, input_size)
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if model_size > config.max_model_params:
        print(f"dropped model which was too large with {model_size} params. Max is: {config.max_model_params}")
        accuracy = 0
    else:
        num_epochs = config.loss_based_stopping_max_epochs \
            if config.loss_based_stopping_in_evolution \
            else config.epochs_in_evolution
        try:
            model = model.to(config.get_device())
            accuracy = evaluate(model, n_epochs=num_epochs)
            if accuracy == "retry":
                raise Exception("no retries in evolution")
        except Exception as e:  # a net that does not fit on a shared GPU scores 0 instead of killing the search
            if not _is_cuda_oom(e):
                raise
            print(f"dropped model that OOM'd with {model_size} params (proc {mp.current_process().name})")
            accuracy = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    blueprint.update_best_sample_map(sample_map, accuracy)
    fitness = [accuracy, model_size]
    blueprint.report_fitness(fitness)
    parse_number = blueprint.n_evaluations

    print("Blueprint - {:^5} - accuracy: {:05.2f}% (proc {}) epochs: {}"
          .format(blueprint.id, accuracy * 100, mp.current_process().name, config.epochs_in_evolution))

    if config.plot_every_genotype:
        blueprint.visualize(parse_number=parse_number,
                            prefix="g" + str(singleton.instance.generation_number) + "_" + str(blueprint.id))

    if config.plot_every_phenotype:
        model.visualize(parse_number=parse_number,
                        prefix="g" + str(singleton.instance.generation_number) + "_" + str(blueprint.id))

    return blueprint
