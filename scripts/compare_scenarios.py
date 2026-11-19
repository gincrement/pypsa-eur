# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Compare high-level results of all scenarios against the base scenario.

For every planning horizon:

- average marginal electricity price in the target countries, i.e. the
  countries with an ``industry: additional_flat_load`` in any scenario. The
  price is the mean over the country's AC buses (consumer buses ``<node> CI``
  excluded), both time-weighted and weighted by the electricity load at these
  buses. The time-weighted shadow price of the consumer bus is reported as
  ``CI price`` where hourly matching is active. The load weighting uses all
  electricity loads at AC and low voltage buses of the country.
- installed capacities per country and carrier (``p_nom_opt``/``s_nom_opt``
  in MW, ``e_nom_opt`` of stores in MWh; links and lines are attributed to
  the country of ``bus0``) and their difference to the base scenario.
  Components of the matched consumer (buses ``<node> CI ...``) are labelled
  ``<carrier> (CI)``; its grid connection links (carriers ``CI ...``) are
  excluded.

Outputs
-------
- ``prices.csv``: (horizon, country, metric) x scenario
- ``capacities.csv``: (horizon, component, unit, country, carrier) x scenario
- ``capacity_difference.csv``: as above, scenario minus base, only rows with
  a difference above ``threshold``
- ``comparison.pdf``: per horizon one page with prices and one page each with
  the capacity differences in the scenario's target countries and in total
  (rows with differences below ``plot_threshold`` are hidden).
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


def is_ci(buses: pd.Index | pd.Series) -> np.ndarray:
    return np.asarray(pd.Series(buses).str.contains(" CI", regex=False))


def electricity_prices(n: pypsa.Network, countries: list[str]) -> pd.DataFrame:
    """
    Average marginal prices per country in EUR/MWh.
    """
    w = n.snapshot_weightings.generators
    buses = n.buses[(n.buses.carrier == "AC") & ~is_ci(n.buses.index)]
    prices = n.buses_t.marginal_price.reindex(columns=buses.index)
    # electricity demand in the country: loads at AC and low voltage buses
    el_buses = n.buses.index[
        n.buses.carrier.isin(["AC", "low voltage"]) & ~is_ci(n.buses.index)
    ]
    loads = n.loads[n.loads.bus.isin(el_buses)]
    load_country = loads.bus.map(n.buses.country)
    demand = (
        n.loads_t.p.reindex(columns=loads.index, fill_value=0.0)
        .T.groupby(load_country)
        .sum()
        .T.mul(w, axis=0)
    )
    ci_buses = n.buses[(n.buses.carrier == "AC") & is_ci(n.buses.index)]

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
        ci = ci_buses.index[ci_buses.country == ct]
        if not ci.empty:
            p_ci = n.buses_t.marginal_price[ci].mean(axis=1)
            row["CI price"] = (p_ci * w).sum() / w.sum()
        result[ct] = row
    return pd.DataFrame(result).T.rename_axis("country").stack()


def installed_capacities(n: pypsa.Network) -> pd.Series:
    """
    Optimal capacities per component, unit, country and carrier.
    """
    country_of = n.buses.country.replace("", "EU").fillna("EU")
    caps = []
    for c, (attr, bus_attr, unit) in CAPACITY_ATTRS.items():
        df = n.static(c)
        # grid connection links of the matched consumer are not real capacity
        df = df[~df.carrier.str.startswith("CI ")]
        if df.empty:
            continue
        carrier = df.carrier.where(~is_ci(df[bus_attr]), df.carrier + " (CI)")
        if c == "Line":
            carrier = pd.Series("AC line", index=df.index)
        country = df[bus_attr].map(country_of).fillna("EU")
        s = df[attr].groupby([country.rename("country"), carrier.rename("carrier")]).sum()
        caps.append(pd.concat({(c, unit): s}, names=["component", "unit"]))
    return pd.concat(caps)


def plot_prices(prices: pd.DataFrame, base: str, horizon, pdf: PdfPages) -> None:
    df = prices.xs("price time-weighted", level="metric")  # country x scenario
    diff = df.sub(df[base], axis=0).drop(columns=base)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(7, 1.2 * len(df.index) + 4), 8), sharex=True
    )
    df.plot.bar(ax=ax1, width=0.85, legend=False)
    ax1.set_ylabel("average marginal price [EUR/MWh]")
    ax1.set_title(f"Average electricity price in target countries ({horizon})")
    diff.plot.bar(
        ax=ax2, width=0.85, legend=False, color=[f"C{i + 1}" for i in range(len(diff.columns))]
    )
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel(f"difference to '{base}' [EUR/MWh]")
    ax2.set_xlabel("")
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.3)
    fig.legend(*ax1.get_legend_handles_labels(), title="scenario", fontsize="small",
               loc="upper right", bbox_to_anchor=(1, 1))
    fig.tight_layout(rect=(0, 0, 0.8, 1))
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

    base = snakemake.params.base_scenario
    meta = snakemake.params.scenarios  # {scenario: {"countries": [...], "target": x}}
    threshold = snakemake.params.threshold
    plot_threshold = snakemake.params.plot_threshold
    countries = sorted({ct for m in meta.values() for ct in m["countries"]})

    prices, capacities = {}, {}
    for fn, (scenario, horizon) in zip(snakemake.input.networks, snakemake.params.runs):
        logger.info(f"Reading {fn}")
        n = pypsa.Network(fn)
        prices[(horizon, scenario)] = electricity_prices(n, countries)
        capacities[(horizon, scenario)] = installed_capacities(n)

    prices = pd.concat(prices, axis=1, names=["horizon", "scenario"])
    prices.index.names = ["country", "metric"]
    prices = prices.stack("horizon", future_stack=True).reorder_levels(
        ["horizon", "country", "metric"]
    ).sort_index()

    capacities = pd.concat(capacities, axis=1, names=["horizon", "scenario"]).fillna(0.0)
    capacities = capacities.stack("horizon", future_stack=True).reorder_levels(
        ["horizon", "component", "unit", "country", "carrier"]
    ).sort_index()

    difference = capacities.sub(capacities[base], axis=0).drop(columns=base)
    difference = difference[(difference.abs() > threshold).any(axis=1)]

    prices.to_csv(snakemake.output.prices)
    capacities.to_csv(snakemake.output.capacities)
    difference.to_csv(snakemake.output.capacity_difference)

    scenarios = [s for s in capacities.columns if s != base]
    with PdfPages(snakemake.output.plot) as pdf:
        for horizon in prices.index.unique("horizon"):
            plot_prices(prices.loc[horizon], base, horizon, pdf)

            diff = difference.loc[horizon] if horizon in difference.index.unique(
                "horizon"
            ) else difference.iloc[:0].droplevel("horizon")

            # difference in each scenario's own target countries
            target = {}
            for s in scenarios:
                cts = meta[s]["countries"]
                d = diff[s][diff.index.get_level_values("country").isin(cts)]
                target[s] = d.groupby(level=["unit", "carrier"]).sum()
            target = pd.DataFrame(target).fillna(0.0)
            target = target[(target.abs() > threshold).any(axis=1)]
            target.index = label(target.index)
            plot_heatmap(
                target,
                f"Capacity difference to '{base}' in the scenario's target country ({horizon})",
                pdf,
                plot_threshold,
            )

            # difference summed over all countries
            total = diff[scenarios].groupby(level=["unit", "carrier"]).sum()
            total = total[(total.abs() > threshold).any(axis=1)]
            total.index = label(total.index)
            plot_heatmap(
                total,
                f"Capacity difference to '{base}', all countries ({horizon})",
                pdf,
                plot_threshold,
            )
