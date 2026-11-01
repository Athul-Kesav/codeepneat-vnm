"""Draw the evolved winners with CoDeepNEAT's own Graphviz visualisers -> figures/architectures/.

For each model: <run>_bp<id>_blueprint_genome.png (BlueprintGenome.visualize) and
<run>_bp<id>_phenotype.png (Network.visualize). Needs the Graphviz `dot` binary on PATH.
"""
import os
import shutil

import recipe  # noqa: F401  (sets cwd / sys.path)
from recipe import setup_config
from configuration import config
from runs import runs_manager
from src.analysis.run import get_run
from src.phenotype.neural_network.evaluator.fully_train import _create_model

OUT = os.path.join(recipe.ROOT, "..", "figures", "architectures")
MODELS = [  # run, dataset (config), [(blueprint, feature multiplier)]
    ("venom_cifar100_res", "cifar100", [(493, 1), (482, 1), (741, 1)]),
    ("venom_cifar100", "cifar100", [(430, 1), (427, 1), (499, 1)]),
    ("venom_cifar10", "cifar10", [(495, 1), (671, 1), (546, 1)]),
    ("ada_cifar10_strong", "cifar10", [(1485, 3), (6826, 3), (6724, 3)]),
    ("imagenet100", "imagenet100", [(207, 1), (182, 1), (230, 1)]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for run_name, ds, models in MODELS:
        setup_config(run_name, ds)
        config.view_graph_plots = False
        run = get_run(run_name)
        best = {}
        for bp, gen in run.get_most_accurate_blueprints(300):
            best.setdefault(bp.id, (bp, gen))
        gdir = runs_manager.get_graphs_folder_path(run_name)
        os.makedirs(gdir, exist_ok=True)
        hw = 128 if ds == "imagenet100" else 32
        for bp_id, fm in models:
            bp, gen = best[bp_id]
            model = _create_model(run, bp, gen, [2, 3, hw, hw], fm)
            before = set(os.listdir(gdir))
            model.visualize(prefix=f"{run_name}_")
            bp.visualize(prefix=f"{run_name}_")
            for f in sorted(set(os.listdir(gdir)) - before):
                if f.endswith(".png"):
                    kind = "phenotype" if "phenotype" in f else "blueprint_genome"
                    shutil.copy(os.path.join(gdir, f), os.path.join(OUT, f"{run_name}_bp{bp_id}_{kind}.png"))
        shutil.rmtree(gdir, ignore_errors=True)
    print("saved to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
