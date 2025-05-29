"""Functions to visualize features."""


def lower_upper_iqr(series, q_lower=0.25, q_upper=0.75):
    ql = series.quantile(q_lower)
    qu = series.quantile(q_upper)
    iqr = qu - ql
    return ql, qu, iqr


def iqr_range(
    series, q_lower: float | None = 0.25, q_upper: float | None = 0.75, r: float = 1.5
):
    if q_lower is None:
        q_lower = 0.0
        ql, qu, iqr = lower_upper_iqr(series, q_lower=q_lower, q_upper=q_upper)
        return ql, ql + r * iqr
    if q_upper is None:
        q_upper = 1.0
        ql, qu, iqr = lower_upper_iqr(series, q_lower=q_lower, q_upper=q_upper)
        return qu - r * iqr, qu

    ql, qu, iqr = lower_upper_iqr(series, q_lower=q_lower, q_upper=q_upper)
    return ql - (r - 1) / 2 * iqr, qu + (r - 1) / 2 * iqr
