"""Polars utilities for unnesting structs."""

from collections.abc import Sequence

import polars as pl


def _unnest_renaming_expression(df: pl.DataFrame, col: str, sep: str = "-") -> pl.Expr:
    return pl.col(col).struct.rename_fields(
        [f"{col}{sep}{f.name}" for f in df.schema.get(col).fields]
    )


def unnest_structs(
    df: pl.DataFrame, cols: str | Sequence[str], sep: str = "-"
) -> pl.DataFrame:
    if isinstance(cols, str):
        cols = (cols,)
    return df.select(
        [
            pl.exclude(cols),
            *[_unnest_renaming_expression(df, col, sep=sep) for col in cols],
        ]
    ).unnest(cols)


def unnest_all_structs(df: pl.DataFrame, sep: str = ".") -> pl.DataFrame:
    cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Struct]
    return unnest_structs(df, cols, sep=sep)
