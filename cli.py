from typing import Literal

import click

import newaresql.local.clone


@click.command()
@click.argument(
    "root",
    default=None,
    help="Root directory to clone the database into.",
)
@click.argument(
    "host",
    default=None,
    help="BTS server host",
)
@click.argument(
    "port",
    default=3306,
    help="BTS server port",
)
@click.argument(
    "user",
    default=None,
    help="BTS server user",
)
@click.argument(
    "password",
    default=None,
    help="BTS server password",
)
@click.argument(
    "database",
    default=None,
    help="Database name",
)
@click.argument(
    "fmt",
    default=None,
    help="Export format, one of (parquet, csv, feather, ipc)",
)
@click.argument(
    "chunksize",
    default=None,
    help="Number of rows to include in each chunk when exporting.",
)
def clone(
    root: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    fmt: Literal["parquet", "csv", "feather", "ipc"],
    chunksize: int,
):
    newaresql.local.clone.clone(
        root=root,
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        fmt=fmt,
        chunksize=chunksize,
    )


if __name__ == "__main__":
    clone()
