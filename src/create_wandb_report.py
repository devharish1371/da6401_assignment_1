"""
Create a W&B Report programmatically for DA6401 Assignment 1.

Usage:
    python src/create_wandb_report.py --project da6401-a1
"""

import wandb
import argparse

try:
    import wandb_workspaces.reports.v2 as wr
except ImportError:
    import wandb.apis.reports as wr


def create_report(project_name, entity=None):
    api = wandb.Api(timeout=60)

    # Resolve entity
    if entity is None:
        entity = api.default_entity
    proj_path = f"{entity}/{project_name}"

    print(f"Connecting to W&B Project: {proj_path}...")
    try:
        all_runs = list(api.runs(proj_path))
        print(f"Found {len(all_runs)} runs in project.")
    except Exception as e:
        print(f"Error accessing W&B project {proj_path}: {e}")
        print("Please ensure you run 'wandb login' and that you have run the experiments first.")
        return

    report = wr.Report(
        project=project_name,
        entity=entity,
        title="DA6401 Assignment 1: MLP Experiments Report",
        description="Auto-generated report encompassing all 10 experiments (2.1 - 2.10) for DA6401.",
    )

    blocks = [
        wr.MarkdownBlock(text=(
            "# DA6401 Assignment 1 - Deep Learning\n\n"
            "This report contains the automated tracking, metrics, and textual "
            "analysis for all 10 mandatory experiments."
        )),
    ]

    # Group runs by experiment prefix
    exp_runs = {}
    for run in all_runs:
        name = run.name
        if name.startswith("2."):
            prefix = name.split("_")[0]  # "2.1", "2.3", etc.
            exp_runs.setdefault(prefix, []).append(run)
        elif run.sweep:
            exp_runs.setdefault("2.2", []).append(run)

    # Section titles
    titles = {
        "2.1": "Data Exploration & Class Distribution (3 Marks)",
        "2.2": "Hyperparameter Sweep (6 Marks)",
        "2.3": "Optimizer Showdown (5 Marks)",
        "2.4": "Vanishing Gradient Analysis (5 Marks)",
        "2.5": "Dead Neuron Investigation (6 Marks)",
        "2.6": "Loss Function Comparison (4 Marks)",
        "2.7": "Global Performance Analysis (4 Marks)",
        "2.8": "Error Analysis (5 Marks)",
        "2.9": "Weight Initialization & Symmetry (7 Marks)",
        "2.10": "Fashion-MNIST Transfer Challenge (5 Marks)",
    }

    for i in range(1, 11):
        prefix = f"2.{i}"
        title = titles.get(prefix, prefix)
        blocks.append(wr.MarkdownBlock(text=f"---\n## Experiment {prefix}: {title}"))

        runs_for_exp = exp_runs.get(prefix, [])
        if not runs_for_exp:
            blocks.append(
                wr.MarkdownBlock(text=(
                    f"> *No runs found for {prefix}. "
                    f"Run `experiment_{prefix.replace('.', '_')}` first.*"
                ))
            )
            continue

        # Pull textual analysis from wandb.summary["analysis"]
        seen = set()
        for run in runs_for_exp:
            text = run.summary.get("analysis", "")
            if text and text not in seen:
                seen.add(text)
                blocks.append(wr.MarkdownBlock(text=text))

        # Panel logic
        if prefix == "2.2":
            # Add Parallel Coordinates for Sweep
            panels = [
                wr.ParallelCoordinatesPlot(
                    columns=[
                        wr.ParallelCoordinatesPlotColumn(metric="config.learning_rate"),
                        wr.ParallelCoordinatesPlotColumn(metric="config.optimizer"),
                        wr.ParallelCoordinatesPlotColumn(metric="config.num_layers"),
                        wr.ParallelCoordinatesPlotColumn(metric="config.activation"),
                        wr.ParallelCoordinatesPlotColumn(metric="val_acc"),
                    ],
                    title="Hyperparameter Relationships"
                ),
                wr.LinePlot(x="Step", y=["val_acc", "train_acc"], title="Sweep Accuracy")
            ]
            runset = wr.Runset(project=project_name, entity=entity)
            runset.query = "Sweep.Name != null" 
        else:
            panels = [
                wr.LinePlot(x="Step", y=["val_acc", "train_acc"], title=f"{prefix} Accuracy"),
                wr.LinePlot(x="Step", y=["val_loss", "train_loss"], title=f"{prefix} Loss"),
            ]
            runset = wr.Runset(project=project_name, entity=entity)
            runset.query = f"Name.startsWith('{prefix}')"

        blocks.append(
            wr.PanelGrid(
                runsets=[runset],
                panels=panels,
            )
        )

    report.blocks = blocks
    try:
        report.save()
        print(f"\n✅ Report generated successfully!")
        print(f"🔗 View your report here: {report.url}")
    except Exception as e:
        print(f"Failed to save report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate W&B Report for DA6401")
    parser.add_argument(
        "--project", type=str, default="da6401-a1", help="W&B Project Name"
    )
    parser.add_argument(
        "--entity", type=str, default=None, help="W&B Entity (Team) Name"
    )
    args = parser.parse_args()

    create_report(args.project, args.entity)
