import click
import os
import pprint
from importlib import import_module
from itertools import permutations

import numpy as np
import pandas as pd
from ruamel import yaml
from ruamel.yaml import YAML
from ruamel.yaml.composer import ComposerError
from sklearn.pipeline import Pipeline

from smp import runs_dir
from smp import submissions_dir
from smp.features.features import Loader


def get_trials(file_name):

    default_data = []
    for file in os.listdir(runs_dir / "libs"):
        with open(runs_dir / "libs" / file) as fp:
            default_data.append(fp.read())

    with open(runs_dir / file_name) as fp:
        user_data = fp.read()

    for perm in permutations(default_data):
        try:
            merged_data = yaml.load(
                "\n".join(list(perm) + [user_data]), Loader=yaml.RoundTripLoader
            )
            break
        except ComposerError as E:
            continue
    else:
        raise ComposerError("Was not able to resolve libraries")

    resolved_yaml = yaml.dump(merged_data, Dumper=yaml.RoundTripDumper)

    my = YAML(typ="safe")
    return my.load(resolved_yaml)["trials"]


def resolve_array(array, parameters):
    return [resolve_cls(item, parameters) for item in array]


def resolve_cls(step, parameters):
    if isinstance(step, (list, tuple)):
        return resolve_array(step, parameters)

    if isinstance(step, dict):

        if "cls" in step:

            *module, cls = step["cls"].split(".")
            cls = getattr(import_module(".".join(module)), cls)
            if "parameters" in step:
                step_params = {
                    name: resolve_cls(p, parameters)
                    for name, p in step["parameters"].items()
                }
                return cls(**step_params)
            else:
                return cls
        elif "$ref" in step:
            return resolve_cls(parameters[step["$ref"]], {})

    return step


def resolve_pipe(trial):
    return Pipeline(
        [
            (step["name"], resolve_cls(step, trial["parameters"]))
            if "name" in step
            else resolve_cls(step, trial["parameters"]).to_step()
            for step in trial["pipeline"]
        ],
        verbose=True,
    )


def run(loader, trial):

    pipe = resolve_pipe(trial)

    X_train = loader.train.iloc[:, :-1]
    y_train = np.log(loader.train.iloc[:, -1] + 1)
    pipe.fit(X_train, y_train)
    scores = pipe.steps[-1][-1].cv_results_
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(f"Mean scores: {(-scores['mean_test_score'])**(1/2)}")
    pp.pprint(f"Best params{pipe.steps[-1][-1].best_params_}")
    pp.pprint(f"Best score: {min((-scores['mean_test_score'])**(1/2))}")

    predictions = pipe.predict(loader.test)

    df = pd.DataFrame(
        {
            "Id": loader.test.index,
            "Predicted": (np.exp(predictions)).round().astype(int),
        },
        dtype=int,
    )

    return df


def main(file_name):

    trials = get_trials(file_name)
    loader = Loader()
    for i, trial in enumerate(trials):
        result_dir = submissions_dir / os.path.splitext(file_name)[0] / str(i)
        os.makedirs(result_dir, exist_ok=True)

        trial_yaml = YAML(typ="safe")
        with open("trials.yml", "w") as f:
            trial_yaml.dump({"trials": [trial]}, f)

        df = run(loader, trial)

        df.to_csv(
            result_dir / f"submission.csv", index=False,
        )


@click.command()
@click.argument("file_name", type=click.Path())
def cli(file_name):
    main(file_name)


if __name__ == "__main__":
    # main("randomforest.yml")
    # main("linear_model.yml")
    main("boostedtrees.yml")

    # cli()
