"""Console script for hubbardml."""
import click

from . import projects


@click.group()
def hml():
    """The HubbardML command line utilities"""


@hml.command()
@click.argument("dataset")
@click.option("-m", "--model", required=True)
@click.option("-p", "--project", default=None)
@click.option("--cutoff", default=projects.Project.DEFAULT_PARAM_CUTOFF, help="Drop all param_out less than this value")
def new(dataset, model, project, cutoff):
    """Create a new project."""
    project = projects.Project.new(dataset, path=project, model_type=model, param_cutoff=cutoff)
    project.save()
    click.echo(f"Path {project.path}, {len(project.dataset)} data points")


@hml.command()
@click.argument("project_path")
@click.option("-n", "--max-iters", default=5_000)
@click.option("-d", "--dev", default=None)
@click.option("--overfit", default=None, type=int)
def train(project_path, max_iters, dev, overfit):
    """Train an existing model"""
    try:
        project = projects.Project(project_path)
        if dev is not None:
            click.echo(f"Switching to {dev}")
            project.to(dev)

        res = project.train(max_iters=max_iters, overfitting_window=overfit)
        click.echo(f"Stop condition: {str(res)}")
    except Exception as exc:
        click.echo(str(exc))
        return 1  # Generic error code
    else:
        project.save()


if __name__ == "__main__":
    hml()
