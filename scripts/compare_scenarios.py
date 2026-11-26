# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Compare high-level results of the country specific scenarios against the base
scenario.

Each scenario is compared against the base only in its own target countries,
i.e. the countries with an ``industry: additional_flat_load`` in that scenario
(e.g. DE scenarios vs base in DE, FR scenarios vs base in FR). Scenarios
without a target country are skipped.

For every planning horizon:

- average marginal electricity price in the target country. The price is the
  mean over the country's AC buses (consumer buses ``<node> CI`` excluded),
  both time-weighted and weighted by the electricity load at these buses. The
  time-weighted price at the bus of the additional flat load (``<node> CI``
  with hourly matching, otherwise the grid node) is reported as ``additional
  load price``; as the base has no additional load, it is compared with the
  base's time-weighted average price in the country. The load
  weighting uses all electricity loads at AC and low voltage buses.
- installed capacities per carrier in the target country (``p_nom_opt``/
  ``s_nom_opt`` in MW, ``e_nom_opt`` of stores in MWh; links and lines are
  attributed to the country of ``bus0``). Components of the matched consumer
  (buses ``<node> CI ...``) are labelled ``<carrier> (CI)``; its grid
  connection links (carriers ``CI ...``) are excluded.

Outputs
-------
All tables have the columns ``value`` (of the scenario), ``base`` and
``difference``.

- ``prices.csv``: (horizon, scenario, country, metric)
- ``capacities.csv``: (horizon, scenario, component, unit, country, carrier)
- ``capacity_difference.csv``: as above, only rows with a difference above
  ``threshold``
- ``comparison.pdf``: per horizon one page with the prices and one page per
  target country with the capacity differences of its scenarios (rows with
  differences below ``plot_threshold`` are hidden).
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.backends.backend_pdf import PdfPages

from scripts._helpers import configure_logging

logger = logging.getLogger(__name__)

CAPACITY_ATTRS = {
    "Generator": ("p_nom_opt", "bus", "MW"),
    "StorageUnit": ("p_nom_opt", "bus", "MW"),
    "Link": ("p_nom_opt", "bus0", "MW"),
    "Line": ("s_nom_opt", "bus0", "MW"),
    "Store": ("e_nom_opt", "bus", "MWh"),
}


ADDITIONAL_LOAD = " additional flat load"
ADDITIONAL_PRICE = "additional load price"
# metric of the base scenario the scenario's metric is compared against;
# the base has no additional load, so its price at the additional bus is
# compared with the base's average price in the country
BASE_METRIC = {ADDITIONAL_PRICE: "price time-weighted"}


def is_ci(buses: pd.Index | pd.Series) -> np.ndarray:
    return np.asarray(pd.Series(buses).str.contains(" CI", regex=False))


def electricity_prices(n: pypsa.Network, countries: list[str]) -> pd.DataFrame:
    """
    Average marginal prices per country in EUR/MWh.
    """
    w = n.snapshot_weightings.generators
    buses = n.c.buses.static[
        (n.c.buses.static.carrier == "AC") & ~is_ci(n.c.buses.static.index)
    ]
    prices = n.c.buses.dynamic.marginal_price.reindex(columns=buses.index)
    # electricity demand in the country: loads at AC and low voltage buses
    el_buses = n.c.buses.static.index[
        n.c.buses.static.carrier.isin(["AC", "low voltage"])
        & ~is_ci(n.c.buses.static.index)
    ]
    loads = n.c.loads.static[n.c.loads.static.bus.isin(el_buses)]
    load_country = loads.bus.map(n.c.buses.static.country)
    demand = (
        n.c.loads.dynamic.p.reindex(columns=loads.index, fill_value=0.0)
        .T.groupby(load_country)
        .sum()
        .T.mul(w, axis=0)
    )
    # buses of the additional flat load: "<node> CI" with hourly matching,
    # otherwise the grid node itself
    add = n.c.loads.static[n.c.loads.static.index.str.endswith(ADDITIONAL_LOAD)]
    add_buses = add.bus.drop_duplicates()
    add_country = add_buses.map(n.c.buses.static.country)

    result = {}
    for ct in countries:
        b = buses.index[buses.country == ct]
        if b.empty:
            continue
        p = prices[b]
        row = {
            "price time-weighted": (p.mean(axis=1) * w).sum() / w.sum(),
            "price load-weighted": (p.mean(axis=1) * demand[ct]).sum()
            / demand[ct].sum()
            if ct in demand
            else np.nan,
        }
        b_add = add_buses[add_country == ct]
        if not b_add.empty:
            p_add = n.c.buses.dynamic.marginal_price[b_add].mean(axis=1)
            row[ADDITIONAL_PRICE] = (p_add * w).sum() / w.sum()
        result[ct] = row
    return pd.DataFrame(result).T.rename_axis("country").stack()


def installed_capacities(n: pypsa.Network) -> pd.Series:
    """
    Optimal capacities per component, unit, country and carrier.
    """
    country_of = n.c.buses.static.country.replace("", "EU").fillna("EU")
    caps = []
    for c, (attr, bus_attr, unit) in CAPACITY_ATTRS.items():
        df = n.c[c].static
        # grid connection links of the matched consumer are not real capacity
        df = df[~df.carrier.str.startswith("CI ")]
        if df.empty:
            continue
        carrier = df.carrier.where(~is_ci(df[bus_attr]), df.carrier + " (CI)")
        if c == "Line":
            carrier = pd.Series("AC line", index=df.index)
        country = df[bus_attr].map(country_of).fillna("EU")
        s = (
            df[attr]
            .groupby([country.rename("country"), carrier.rename("carrier")])
            .sum()
        )
        caps.append(pd.concat({(c, unit): s}, names=["component", "unit"]))
    return pd.concat(caps)


def country_group(meta: dict, scenario: str) -> str:
    return "+".join(meta[scenario]["countries"]) or "none"


def compare_to_base(
    df: pd.DataFrame, base: str, meta: dict, scenarios: list[str]
) -> pd.DataFrame:
    """
    Compare each scenario against the base only in its own target countries.

    ``df`` has a ``country`` index level and one column per scenario. Returns
    a frame indexed by ``scenario`` plus the levels of ``df`` with the columns
    ``value`` (scenario), ``base`` and ``difference``.
    """
    countries = df.index.get_level_values("country")
    out = {}
    for s in scenarios:
        d = df[countries.isin(meta[s]["countries"])]
        out[s] = pd.DataFrame(
            {"value": d[s], "base": d[base], "difference": d[s] - d[base]}
        )
    return pd.concat(out, names=["scenario"])


def plot_prices(prices: pd.DataFrame, meta: dict, horizon, pdf: PdfPages) -> None:
    """
    Average price in each scenario's target country and price at the bus of
    the additional load, next to the base's average price there.
    """
    avg = prices.xs("price time-weighted", level="metric")  # (scenario, country)
    if avg.empty:
        return
    if ADDITIONAL_PRICE in prices.index.unique("metric"):
        add = prices.xs(ADDITIONAL_PRICE, level="metric").reindex(avg.index)
    else:
        logger.warning(f"No '{ADDITIONAL_PRICE}' found; is the additional load added?")
        add = pd.DataFrame(np.nan, index=avg.index, columns=avg.columns)
    order = sorted(avg.index, key=lambda i: (country_group(meta, i[0]), i))
    avg, add = avg.loc[order], add.loc[order]
    xlabels = [f"{s}\n({ct})" for s, ct in avg.index]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(7, 0.7 * len(avg) + 3), 8), sharex=True
    )
    pd.DataFrame(
        {
            "base: country average": avg["base"],
            "scenario: additional load bus": add["value"],
        }
    ).set_axis(xlabels).plot.bar(ax=ax1, width=0.85, color=["0.6", "C3"])
    ax1.set_ylabel("time-weighted marginal price [EUR/MWh]")
    ax1.set_title(f"Electricity price in the scenario's target country ({horizon})")
    ax1.legend(fontsize="small")
    pd.DataFrame(
        {
            "country average": avg["difference"],
            "additional load bus": add["difference"],
        }
    ).set_axis(xlabels).plot.bar(ax=ax2, width=0.7, color=["C0", "C3"])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("difference to base country average [EUR/MWh]")
    ax2.set_xlabel("")
    ax2.legend(fontsize="small")
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame, title: str, pdf: PdfPages, threshold: float = 1.0
) -> None:
    df = df[(df.abs() >= threshold).any(axis=1)]
    if df.empty:
        return
    vmax = df.abs().max().max() or 1.0
    fig, ax = plt.subplots(
        figsize=(1.0 * len(df.columns) + 4, 0.28 * len(df.index) + 2)
    )
    im = ax.imshow(df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(df.columns)), df.columns, rotation=90)
    ax.set_yticks(range(len(df.index)), df.index)
    if df.size <= 600:
        for (i, j), v in np.ndenumerate(df.values):
            if abs(v) > 0:
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="difference to base")
    ax.set_title(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def label(index: pd.MultiIndex) -> pd.Index:
    return pd.Index([f"{carrier} [{unit}]" for unit, carrier in index])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("compare_scenarios")

    configure_logging(snakemake)

    # The duals of the transmission loss tangent constraints (Line mu_pos /
    # mu_neg) are stored with an extra segment dimension, which PyPSA cannot
    # map back to the lines on import and reports once per network. They are
    # not used here, so silence this specific warning.
    logging.getLogger("pypsa.network.io").addFilter(
        lambda record: (
            not (
                "of Line are not in main components dataframe" in record.getMessage()
                and ("mu_pos" in record.getMessage() or "mu_neg" in record.getMessage())
            )
        )
    )

    base = snakemake.params.base_scenario
    meta = snakemake.params.scenarios  # {scenario: {"countries": [...], "target": x}}
    threshold = snakemake.params.threshold
    plot_threshold = snakemake.params.plot_threshold
    # only scenarios with a target country can be compared against the base
    scenarios = [s for s in meta if s != base and meta[s]["countries"]]
    countries = sorted({ct for s in scenarios for ct in meta[s]["countries"]})

    prices, capacities = {}, {}
    for fn, (scenario, horizon) in zip(snakemake.input.networks, snakemake.params.runs):
        logger.info(f"Reading {fn}")
        n = pypsa.Network(fn)
        prices[(horizon, scenario)] = electricity_prices(n, countries)
        capacities[(horizon, scenario)] = installed_capacities(n)

    prices = pd.concat(prices, axis=1, names=["horizon", "scenario"])
    prices.index.names = ["country", "metric"]
    prices = prices.stack("horizon", future_stack=True)
    for metric, base_metric in BASE_METRIC.items():
        rows = prices.index[prices.index.get_level_values("metric") == metric]
        src = pd.MultiIndex.from_tuples(
            [(ct, base_metric, h) for ct, _, h in rows], names=rows.names
        )
        prices.loc[rows, base] = prices[base].reindex(src).values
    prices = (
        compare_to_base(prices, base, meta, scenarios)
        .reorder_levels(["horizon", "scenario", "country", "metric"])
        .sort_index()
        .dropna(how="all")
    )

    capacities = pd.concat(capacities, axis=1, names=["horizon", "scenario"]).fillna(
        0.0
    )
    capacities = capacities.stack("horizon", future_stack=True)
    capacities = (
        compare_to_base(capacities, base, meta, scenarios)
        .reorder_levels(
            ["horizon", "scenario", "component", "unit", "country", "carrier"]
        )
        .sort_index()
    )
    capacities = capacities[(capacities[["value", "base"]] != 0).any(axis=1)]
    difference = capacities[capacities["difference"].abs() > threshold]

    prices.to_csv(snakemake.output.prices)
    capacities.to_csv(snakemake.output.capacities)
    difference.to_csv(snakemake.output.capacity_difference)

    groups = {}
    for s in scenarios:
        groups.setdefault(country_group(meta, s), []).append(s)

    with PdfPages(snakemake.output.plot) as pdf:
        for horizon in prices.index.unique("horizon"):
            plot_prices(prices.loc[horizon], meta, horizon, pdf)

            diff = (
                difference.loc[horizon, "difference"]
                if horizon in difference.index.unique("horizon")
                else pd.Series(dtype=float)
            )
            if diff.empty:
                continue
            # one page per target country: its scenarios vs base in that country
            diff = (
                diff.groupby(level=["scenario", "unit", "carrier"])
                .sum()
                .unstack("scenario")
            )
            for group, members in groups.items():
                d = diff.reindex(columns=members).dropna(how="all").fillna(0.0)
                if d.empty:
                    continue
                d.index = label(d.index)
                plot_heatmap(
                    d,
                    f"Capacity difference to '{base}' in {group} ({horizon})",
                    pdf,
                    plot_threshold,
                )
