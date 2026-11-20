# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# matching granularity -> (pandas period alias, nominal period length in h)
MATCHING_PERIODS = {
    "hourly": ("h", 1),
    "daily": ("D", 24),
    "weekly": ("W", 168),
    "monthly": ("M", 672),
    "annual": ("Y", 8760),
}


def matching_periods(snapshots: pd.Index, mode: str) -> pd.Series:
    """
    Map each snapshot to the calendar period (hour, day, week, month, year)
    in which it starts. Works for sub-hourly, hourly and aggregated snapshots
    and for multi-investment snapshots (periods are kept separate per
    investment period).
    """
    freq = MATCHING_PERIODS[mode][0]
    if isinstance(snapshots, pd.MultiIndex):
        ts = pd.DatetimeIndex(snapshots.get_level_values(-1))
        keys = (
            snapshots.get_level_values(0).astype(str)
            + "|"
            + ts.to_period(freq).astype(str)
        )
    else:
        keys = pd.DatetimeIndex(snapshots).to_period(freq).astype(str)
    codes, labels = pd.factorize(keys)
    return pd.Series(labels[codes], index=snapshots)


def add_hourly_matching_constraints(n, snapshots):
    """
    Hourly matching of the consumers set up by ``add_additional_flat_load``
    in ``prepare_sector_network.py`` (settings in ``n.meta["hourly_matching"]``).

    The consumer bus ``<node> CI`` is supplied by dedicated, newly built VRE
    and storage, by a grid import link (unmatched), optionally by an import
    link for grid VRE that would otherwise be curtailed (matched) and can
    export excess to the grid. Within each matching period p (``mode``:
    hourly, daily, weekly, monthly or annual calendar periods; snapshots are
    assigned to the period they start in, so sub-hourly data is summed per
    clock hour), excess exports offset imports, i.e. the grid acts as a bank
    within but not across periods. With ``target_scope``

    - ``period``: in every period
      sum_{t in p} w_t (import_t - export_t) <= (1 - target) sum_{t in p} w_t load_t
    - ``annual``: unmatched_p >= sum_{t in p} w_t (import_t - export_t),
      unmatched_p >= 0 and sum_p unmatched_p <= (1 - target) sum_t w_t load_t
      (share of consumption matched at the granularity of the periods,
      as in 24/7 CFE scores)

    In every snapshot the curtailment import is limited to the consumer's
    load and to the curtailment of grid VRE (``curtailment_carriers``) in the
    consumer's country:

        curt_import_t <= sum_g (p_max_pu_g,t * p_nom_g - p_g,t)

    Also adds an optional cap on excess exports and makes the dedicated VRE
    share the land potential of the grid VRE at the same node.
    """
    cfg = n.meta.get("hourly_matching")
    if not cfg:
        return

    m = n.model
    w = n.snapshot_weightings.generators.loc[snapshots]
    p_link = m["Link-p"]
    p_nom_gen = m["Generator-p_nom"]
    p_gen = m["Generator-p"]
    ext_gen = n.c.generators.static.index[n.c.generators.static.p_nom_extendable]
    bus_country = n.c.generators.static.bus.map(n.c.buses.static.country)

    for ci in cfg["nodes"]:
        load_i = n.c.loads.static.index[n.c.loads.static.bus == ci]
        load = n.get_switchable_as_dense("Load", "p_set", snapshots)[load_i].sum(axis=1)
        imp = p_link.sel(name=f"{ci} grid import")
        exp = p_link.sel(name=f"{ci} grid export")
        target = cfg["target"]

        mode, scope = cfg["mode"], cfg.get("target_scope", "period")
        periods = matching_periods(snapshots, mode)
        if w.max() > MATCHING_PERIODS[mode][1]:
            logger.warning(
                f"Snapshots ({w.max():g} h) are coarser than the {mode} matching "
                "period; snapshots are assigned to the period they start in."
            )
        group = xr.DataArray(
            periods.values,
            coords={"snapshot": imp.coords["snapshot"]},
            dims="snapshot",
            name="period",
        )
        w_da = xr.DataArray(w.values, coords={"snapshot": imp.coords["snapshot"]})
        net_import = ((imp - exp) * w_da).groupby(group).sum()
        consumption = (load * w).groupby(periods.values).sum()
        consumption.index.name = "period"

        if scope == "period":
            m.add_constraints(
                net_import <= (1 - target) * consumption.to_xarray(),
                name=f"hourly_matching-{ci}",
            )
        else:
            unmatched = m.add_variables(
                lower=0,
                coords=[consumption.index],
                name=f"hourly_matching_unmatched-{ci}",
            )
            m.add_constraints(
                net_import - unmatched <= 0,
                name=f"hourly_matching_unmatched-{ci}",
            )
            m.add_constraints(
                unmatched.sum() <= (1 - target) * consumption.sum(),
                name=f"hourly_matching-{ci}",
            )

        if (share := cfg.get("max_excess_share")) is not None:
            m.add_constraints(
                (exp * w).sum() <= share * (load * w).sum(),
                name=f"hourly_matching_excess-{ci}",
            )

        curt = f"{ci} curtailment import"
        if curt in n.c.links.static.index:
            load_da = load.rename_axis("snapshot").to_xarray()
            m.add_constraints(
                p_link.sel(name=curt) <= load_da,
                name=f"hourly_matching_curtailment_load-{ci}",
            )
            ct = n.c.buses.static.at[ci, "country"]
            gens = n.c.generators.static.index[
                (bus_country == ct)
                & n.c.generators.static.carrier.isin(cfg["curtailment_carriers"])
                & ~n.c.generators.static.bus.isin(cfg["nodes"])
            ]
            p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", snapshots)[
                gens
            ]
            p_max_pu.index.name, p_max_pu.columns.name = "snapshot", "name"
            ext, fix = gens.intersection(ext_gen), gens.difference(ext_gen)
            # curt_import + dispatch - p_max_pu * p_nom(ext) <= p_max_pu * p_nom(fix)
            lhs = p_link.sel(name=curt) + p_gen.sel(name=gens).sum("name")
            if not ext.empty:
                lhs -= (xr.DataArray(p_max_pu[ext]) * p_nom_gen.sel(name=ext)).sum(
                    "name"
                )
            rhs = (p_max_pu[fix] * n.c.generators.static.p_nom[fix]).sum(axis=1)
            rhs.index.name = "snapshot"
            m.add_constraints(
                lhs <= rhs.to_xarray(), name=f"hourly_matching_curtailment-{ci}"
            )

        # shared land potential with grid generators at the same node
        ci_gens = n.c.generators.static.index[
            (n.c.generators.static.bus == ci)
            & n.c.generators.static.index.str.endswith(" CI")
        ]
        for g in ci_gens:
            parent = g.removesuffix(" CI")
            p_max = n.c.generators.static.at[parent, "p_nom_max"]
            if not np.isfinite(p_max) or parent not in ext_gen:
                continue
            m.add_constraints(
                p_nom_gen.sel(name=g) + p_nom_gen.sel(name=parent) <= p_max,
                name=f"hourly_matching_potential-{g}",
            )

        logger.info(
            f"Added hourly matching constraint for {ci} "
            f"(target {target:.0%}, mode {mode}, scope {scope})."
        )


def custom_extra_functionality(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints.
    """
    add_hourly_matching_constraints(n, snapshots)
