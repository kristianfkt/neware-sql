import click

import newaresql.clone
import newaresql.explore
import newaresql.export


@click.group()
def cli():
    pass


@cli.command()
def clone():
    newaresql.clone.clone()


@cli.command()
def export():
    newaresql.export.export()


@cli.command()
def explore():
    newaresql.explore.explore()


if __name__ == "__main__":
    cli()
