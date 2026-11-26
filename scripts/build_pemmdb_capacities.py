# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Translate PEMMDB capacities into fixed network capacities.

Reads PEMMDB (ENTSO-E Pan-European Market Modelling Database) installed
capacities per market node and technology, maps them onto the components of
the composed network and writes them in the format of
`solving.options.fixed_capacities` (columns `component`, `name`,
`capacity`), which rule `solve_network` applies before a dispatch-only
optimisation.

Input (`solving.options.fixed_capacities.pemmdb.path`):

- a directory with the ENTSO-E node files
  ``PEMMDB_<node>[_OFF]_NationalTrends_<year>.xlsx`` (sheets Thermal, Wind,
  Solar, Hydro, Battery, Electrolyser, Other RES, Other Non-RES, DSR), or
- a CSV with columns ``node``, ``technology``, ``capacity`` (MW) and
  optional ``energy`` (MWh).

Thermal technologies are named ``<fuel> - <type>``, e.g. ``Hard coal - old 1``
or ``Gas - CCGT present 2``, so each plant class can be mapped to its own
network component (`sector.conventional_generation_classes`).

Mapping:

- Nodes are mapped to regions of the network (AC bus name without the
  ``AC`` suffix), so that both country (`clustering.administrative.level: 0`)
  and bidding zone (`level: bz`) networks are supported: first by
  `node_map`, then by the region of the same name, then to all regions of
  the node's country (e.g. offshore nodes such as ``DEKF``).
- Technologies are mapped by the first matching regular expression in
  `technology_map` to targets. A target matches components named
  ``<bus> <target>`` or with carrier ``<target>``; if none exists, the
  carrier of its first word is used (e.g. ``coal old 1`` -> ``coal``).
- If several components match (several buses, resource classes, carriers),
  the capacity is split proportionally to their capacity in the composed
  network, otherwise to `p_nom_max`, otherwise equally.
- Capacities are given in MW electric. Links feeding an AC bus (power
  plants, discharger) are converted to `p_nom` at their input (bus0) by
  their efficiency; links consuming electricity (electrolysers, chargers)
  keep the value.
- Battery technologies set charger, discharger and store (energy defaults
  to `electricity.max_hours` x power). For storage units, `energy` sets
  `max_hours`.
- With `zero_missing`, components of assigned carriers in countries covered
  by PEMMDB but without a PEMMDB value are set to zero.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from openpyxl import load_workbook

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

NOM = {
    "Generator": "p_nom",
    "StorageUnit": "p_nom",
    "Link": "p_nom",
    "Store": "e_nom",
}

STORES = ["battery", "home battery"]

COUNTRY_ALIASES = {"UK": "GB", "EL": "GR"}


# ---------------------------------------------------------------- reading


def _rows(wb, sheet: str, max_row: int = 60, max_col: int = 40) -> list[tuple]:
    if sheet not in wb.sheetnames:
        return []
    return list(wb[sheet].iter_rows(max_row=max_row, max_col=max_col, values_only=True))


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _labelled(rows: list[tuple]) -> dict[str, float]:
    """
    Values of sheets with a label in column A and the value in column B.
    """
    return {
        str(r[0]).strip(): _num(r[1])
        for r in rows
        if r and isinstance(r[0], str) and len(r) > 1
    }


def read_thermal(wb) -> list[tuple]:
    out, fuel = [], None
    for r in _rows(wb, "Thermal", max_row=200, max_col=7):
        if len(r) < 7 or r[6] != "number of must run units":
            continue
        fuel = str(r[0]).strip() if r[0] is not None else fuel
        kind = str(r[1]).strip() if r[1] is not None else ""
        tech = f"{fuel} - {kind}" if kind else fuel
        out.append((tech, _num(r[2]), np.nan))
    return out


def read_wind(wb) -> list[tuple]:
    v = _labelled(_rows(wb, "Wind", max_col=2))
    get = lambda key: 1e3 * next((x for k, x in v.items() if k.startswith(key)), 0.0)  # noqa: E731
    return [
        ("Wind Onshore", get("Installed capacities Onshore wind Total"), np.nan),
        ("Wind Offshore", get("Installed capacities Offshore wind Total"), np.nan),
    ]


def read_solar(wb) -> list[tuple]:
    v = _labelled(_rows(wb, "Solar", max_col=2))
    pv = 1e3 * v.get("Installed capacities Photovoltaic (GW):", 0.0)
    rooftop = 1e3 * v.get("Installed capacities Rooftop (GW):", 0.0)
    thermal = 1e3 * (
        v.get("Installed capacities Thermal Solar (GW):", 0.0)
        + v.get("Installed capacities Solar Thermal with Storage (GW):", 0.0)
    )
    return [
        ("Solar PV", pv - rooftop, np.nan),
        ("Solar PV Rooftop", rooftop, np.nan),
        ("Solar Thermal", thermal, np.nan),
    ]


def read_hydro(wb) -> list[tuple]:
    v = _labelled(_rows(wb, "Hydro", max_col=2))

    def get(prefix: str, what: str) -> float:
        key = next((k for k in v if k.startswith(prefix) and what in k), None)
        return v[key] if key is not None else 0.0

    out = [("Hydro Run of River", get("Run of River", "turbining"), np.nan)]
    for prefix, tech in [
        ("Pondage", "Hydro Pondage"),
        ("Reservoir", "Hydro Reservoir"),
        ("Pump Storage (open loop", "Hydro Pump Storage Open Loop"),
        ("Pure Pump Storage", "Hydro Pump Storage Closed Loop"),
    ]:
        out.append(
            (
                tech,
                get(prefix, "turbining"),
                1e3 * get(prefix, "reservoir capacity (GWh)"),
            )
        )
    return out


def read_battery(wb) -> list[tuple]:
    out = []
    for r in _rows(wb, "Battery", max_col=5):
        if (
            r
            and isinstance(r[0], str)
            and r[0].startswith("Battery ")
            and r[0] != "Battery Total"
        ):
            out.append((r[0].strip(), _num(r[2]), _num(r[4])))
    return out


def read_electrolyser(wb) -> list[tuple]:
    total = sum(
        _num(r[2]) or 0.0
        for r in _rows(wb, "Electrolyser", max_col=3)
        if r and isinstance(r[0], str) and r[0].startswith("Electrolyser")
    )
    return [("Electrolyser", total, np.nan)]


def read_other_res(wb) -> list[tuple]:
    rows = _rows(wb, "Other RES", max_row=10, max_col=9)
    names = next((r for r in rows if r and r[4] == "Small Biomass"), None)
    values = next((r for r in rows if r and r[3] == "Installed capacity (MW):"), None)
    if names is None or values is None:
        return []
    return [
        (f"Other RES {names[i]}", _num(values[i]), np.nan)
        for i in range(4, len(names))
        if names[i]
    ]


def read_other_non_res(wb) -> list[tuple]:
    rows = _rows(wb, "Other Non-RES", max_row=10, max_col=60)
    bands = next((r for r in rows if r and r[2] == "Price Band 1"), None)
    caps = next(
        (
            r
            for r in rows
            if r and isinstance(r[1], str) and r[1].startswith("Installed capacity")
        ),
        None,
    )
    if bands is None or caps is None:
        return []
    # each band spans several identical columns
    seen = {}
    for band, cap in zip(bands[2:], caps[2:]):
        if band and band not in seen:
            seen[band] = _num(cap)
    return [("Other Non-RES", np.nansum(list(seen.values())), np.nan)]


def read_dsr(wb) -> list[tuple]:
    rows = _rows(wb, "DSR", max_row=12, max_col=20)
    caps = next((r for r in rows if r and r[1] == "Capacity"), None)
    if caps is None:
        return []
    return [("DSR", np.nansum([_num(x) for x in caps[2:]]), np.nan)]


READERS = [
    read_thermal,
    read_wind,
    read_solar,
    read_hydro,
    read_battery,
    read_electrolyser,
    read_other_res,
    read_other_non_res,
    read_dsr,
]


def read_pemmdb_dir(path: str) -> pd.DataFrame:
    files = sorted(
        f for f in Path(path).glob("PEMMDB_*.xlsx") if not f.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(f"No PEMMDB files PEMMDB_*.xlsx in {path}.")
    records = []
    for f in files:
        node = re.match(r"PEMMDB_([A-Z0-9]+?)(_OFF)?_", f.name)
        if node is None:
            logger.warning(f"Skipping {f.name}: no market node in file name.")
            continue
        node = node.group(1)
        wb = load_workbook(f, read_only=True, data_only=True)
        try:
            for reader in READERS:
                records += [(node, *rec) for rec in reader(wb)]
        finally:
            wb.close()
    logger.info(f"Read {len(files)} PEMMDB files from {path}.")
    df = pd.DataFrame(records, columns=["node", "technology", "capacity", "energy"])
    return df.groupby(["node", "technology"], as_index=False, sort=False).sum(
        min_count=1
    )


def read_pemmdb(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        available = sorted(p.name for p in Path(path).parent.glob("*"))
        raise FileNotFoundError(
            f"PEMMDB data {path} not found (available: {available}). Map the "
            "horizon to an available PEMMDB year via `year_map`."
        )
    if Path(path).is_dir():
        df = read_pemmdb_dir(path)
    else:
        df = pd.read_csv(path, comment="#")
        df.columns = df.columns.str.strip().str.lower()
        missing = {"node", "technology", "capacity"}.difference(df.columns)
        if missing:
            raise ValueError(f"PEMMDB file {path} lacks columns {missing}.")
    if "energy" not in df.columns:
        df["energy"] = np.nan
    df["node"] = df["node"].astype(str).str.strip()
    df["technology"] = df["technology"].astype(str).str.strip()
    df[["capacity", "energy"]] = df[["capacity", "energy"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return df


# ---------------------------------------------------------------- mapping


def ac_regions(n: pypsa.Network) -> pd.DataFrame:
    """
    AC buses with their region code (bus name without the ``AC`` suffix).
    """
    buses = n.buses.query("carrier == 'AC'")[["country"]].copy()
    buses["region"] = buses.index.str.replace(r"\s*AC$", "", regex=True)
    return buses


def map_nodes(nodes: pd.Index, buses: pd.DataFrame, node_map: dict) -> pd.Series:
    """
    Map PEMMDB nodes to lists of AC buses.
    """
    by_region = pd.Series(buses.index, index=buses.region)
    mapping, fallback = {}, []
    for node in nodes:
        regions = [r for r in node_map.get(node, []) if r in by_region.index]
        if not regions and node in by_region.index:
            regions = [node]
        if regions:
            mapping[node] = by_region.loc[regions].tolist()
            continue
        country = COUNTRY_ALIASES.get(node[:2], node[:2])
        mapping[node] = buses.index[buses.country == country].tolist()
        fallback.append(node)
    if fallback:
        logger.info(f"PEMMDB nodes mapped to all regions of their country: {fallback}")
    return pd.Series(mapping, dtype=object)


def map_technologies(technologies: pd.Index, technology_map: dict) -> pd.Series:
    """
    Map PEMMDB technologies to lists of targets (first matching pattern).
    """
    patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in technology_map.items()]
    mapping = {}
    for tech in technologies:
        mapping[tech] = None
        for pattern, targets in patterns:
            m = pattern.search(tech)
            if m:
                mapping[tech] = [m.expand(t) for t in targets]
                break
    return pd.Series(mapping, dtype=object)


def component_locations(n: pypsa.Network, c: str) -> pd.Series:
    """
    AC bus of each component (via bus location; for links bus1, else bus0).
    """
    static = n.c[c].static
    loc = n.buses.location.where(n.buses.location != "", n.buses.index)
    if c == "Link":
        bus = static.bus1.map(loc)
        return bus.where(
            bus.isin(n.buses.index[n.buses.carrier == "AC"]), static.bus0.map(loc)
        )
    return static.bus.map(loc)


def output_factor(n: pypsa.Network, c: str, name: str, ac: pd.Index) -> float:
    """
    Electrical MW per unit of component capacity.

    Links feeding an AC bus are sized at their input, so their electrical
    capacity is ``p_nom * efficiency``.
    """
    if c == "Link" and n.links.at[name, "bus1"] in ac:
        return n.links.at[name, "efficiency"]
    return 1.0


def electric_buses(n: pypsa.Network) -> pd.Index:
    """
    Buses whose output counts as electrical capacity (AC and low voltage).
    """
    return n.buses.index[n.buses.carrier.isin(["AC", "low voltage"])]


def candidates(
    n: pypsa.Network, targets: list[str], buses: list[str], locs: dict
) -> list:
    """
    Components matching the targets at the buses.
    """
    found = []
    for target in targets:
        match = []
        for c in ["Generator", "StorageUnit", "Link"]:
            static = n.c[c].static
            if static.empty:
                continue
            at_bus = locs[c].isin(buses)
            by_name = static.index.isin([f"{b} {target}" for b in buses])
            by_carrier = static.carrier == target
            sel = static.index[at_bus & (by_name | by_carrier)]
            match += [(c, name) for name in sel]
        if not match:
            base = target.split(" ")[0]
            for c in ["Generator", "StorageUnit", "Link"]:
                static = n.c[c].static
                if static.empty:
                    continue
                sel = static.index[locs[c].isin(buses) & (static.carrier == base)]
                match += [(c, name) for name in sel]
        found += match
    return list(dict.fromkeys(found))


def split(n: pypsa.Network, comps: list, total: float, ac: pd.Index) -> pd.Series:
    """
    Split electrical capacity onto components, returns component capacities.
    """
    idx = pd.MultiIndex.from_tuples(comps, names=["component", "name"])
    current = pd.Series([n.c[c].static.at[m, NOM[c]] for c, m in comps], idx, float)
    potential = pd.Series(
        [n.c[c].static.at[m, "p_nom_max"] for c, m in comps], idx, float
    )
    factor = pd.Series([output_factor(n, c, m, ac) for c, m in comps], idx, float)
    if (current * factor).sum() > 0:
        weights = current * factor
    elif np.isfinite(potential).all() and potential.sum() > 0:
        weights = potential * factor
    else:
        weights = pd.Series(1.0, index=idx)
    return total * weights / weights.sum() / factor


def storage(
    n: pypsa.Network, kind: str, buses: list[str], power: float, energy: float
) -> list:
    """
    Charger, discharger and store of battery type ``kind`` at the buses.
    """
    loc = n.buses.location.where(n.buses.location != "", n.buses.index)
    chargers = n.links.index[
        (n.links.carrier == f"{kind} charger") & n.links.bus0.map(loc).isin(buses)
    ]
    rows = []
    for charger in chargers:
        s = 1 / len(chargers)
        store_bus = n.links.at[charger, "bus1"]
        rows.append(("Link", charger, power * s))
        dis = n.links.index[
            (n.links.carrier == f"{kind} discharger") & (n.links.bus0 == store_bus)
        ]
        rows += [("Link", d, power * s / n.links.at[d, "efficiency"]) for d in dis]
        st = n.stores.index[(n.stores.carrier == kind) & (n.stores.bus == store_bus)]
        rows += [("Store", x, energy * s) for x in st]
    return rows


def build_capacities(
    n: pypsa.Network, pemmdb: pd.DataFrame, config: dict, max_hours: dict
) -> pd.DataFrame:
    buses = ac_regions(n)
    ac = electric_buses(n)
    locs = {c: component_locations(n, c) for c in ["Generator", "StorageUnit", "Link"]}

    node_buses = map_nodes(
        pd.Index(pemmdb.node.unique()), buses, config.get("node_map") or {}
    )
    tech_targets = map_technologies(
        pd.Index(pemmdb.technology.unique()), config["technology_map"]
    )
    pemmdb = pemmdb.assign(
        buses=pemmdb.node.map(node_buses), targets=pemmdb.technology.map(tech_targets)
    )

    unmapped = pemmdb[pemmdb.targets.isna()]
    if not unmapped.empty:
        logger.warning(
            "PEMMDB technologies without mapping, ignored [MW]:\n"
            + unmapped.groupby("technology").capacity.sum().to_string()
        )
    ignored = pemmdb[pemmdb.targets.map(lambda t: t == [])]
    if not ignored.empty:
        logger.info(
            "PEMMDB technologies ignored by technology_map [MW]:\n"
            + ignored.groupby("technology").capacity.sum().round(0).to_string()
        )
    no_bus = pemmdb[pemmdb.buses.map(len) == 0]
    if not no_bus.empty:
        logger.warning(
            "PEMMDB nodes without bus in the network, ignored [MW]:\n"
            + no_bus.groupby("node").capacity.sum().round(0).to_string()
        )

    df = pemmdb[pemmdb.targets.map(bool) & (pemmdb.buses.map(len) > 0)].copy()
    df["buses"] = df.buses.map(tuple)
    df["targets"] = df.targets.map(tuple)
    grouped = df.groupby(["buses", "targets"], sort=False)[["capacity", "energy"]].sum(
        min_count=1
    )

    rows, storage_energy, unassigned = [], {}, []
    for (bus_list, targets), (power, energy) in grouped.iterrows():
        bus_list, targets = list(bus_list), list(targets)
        power = 0.0 if np.isnan(power) else power
        kinds = [t for t in targets if t in STORES]
        if kinds:
            for kind in kinds:
                e = (
                    energy
                    if not np.isnan(energy)
                    else power * max_hours.get("battery", 2.0)
                )
                rows_kind = storage(n, kind, bus_list, power, e)
                if not rows_kind and power > 0:
                    unassigned.append((kind, bus_list, power))
                rows += rows_kind
            continue
        comps = candidates(n, targets, bus_list, locs)
        if not comps:
            if power > 0:
                unassigned.append((targets, bus_list, power))
            continue
        caps = split(n, comps, power, ac)
        rows += [(c, m, cap) for (c, m), cap in caps.items()]
        if not np.isnan(energy):
            for (c, m), cap in caps.items():
                if c == "StorageUnit":
                    storage_energy[m] = storage_energy.get(m, 0.0) + (
                        energy * cap / power if power else 0.0
                    )

    if unassigned:
        msg = "\n".join(f"  {t} at {b}: {p:.0f} MW" for t, b, p in unassigned)
        logger.warning(f"PEMMDB capacities without network component:\n{msg}")

    out = pd.DataFrame(rows, columns=["component", "name", "capacity"])
    out = out.groupby(["component", "name"], as_index=False, sort=False).sum()

    static = {c: n.c[c].static for c in NOM}
    carrier_of = lambda c, m: static[c].at[m, "carrier"]  # noqa: E731

    if config.get("zero_missing", True) and not out.empty:
        covered = buses.index[
            buses.country.isin(buses.loc[[b for bl in df.buses for b in bl], "country"])
        ]
        assigned = {carrier_of(c, m) for c, m in zip(out.component, out.name)}
        done = set(zip(out.component, out.name))
        zero = []
        for c in ["Generator", "StorageUnit", "Link"]:
            s = static[c]
            sel = s.index[s.carrier.isin(assigned) & locs[c].isin(covered)]
            zero += [(c, m) for m in sel if (c, m) not in done]
        sel = n.stores.index[
            n.stores.carrier.isin(assigned)
            & n.stores.bus.map(n.buses.location).isin(covered)
        ]
        zero += [("Store", m) for m in sel if ("Store", m) not in done]
        if zero:
            logger.info(
                f"Set {len(zero)} components without PEMMDB value to zero (zero_missing)."
            )
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(zero, columns=["component", "name"]).assign(
                        capacity=0.0
                    ),
                ],
                ignore_index=True,
            )

    out["max_hours"] = np.nan
    for name, energy in storage_energy.items():
        i = out.index[(out.component == "StorageUnit") & (out.name == name)]
        cap = out.loc[i, "capacity"]
        out.loc[i, "max_hours"] = (energy / cap).where(cap > 0)

    out["carrier"] = [carrier_of(c, m) for c, m in zip(out.component, out.name)]
    out["bus"] = [
        locs[c].at[m] if c in locs else n.buses.location.get(static[c].at[m, "bus"])
        for c, m in zip(out.component, out.name)
    ]
    out["capacity_el"] = [
        cap * output_factor(n, c, m, ac)
        for c, m, cap in zip(out.component, out.name, out.capacity)
    ]

    summary = out.query("component != 'Store'").groupby("carrier").capacity_el.sum()
    logger.info(
        "Assigned electrical capacity per carrier [GW]:\n"
        + (summary / 1e3).round(1).to_string()
    )
    return out


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_pemmdb_capacities", horizon=2030)

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    pemmdb_path = snakemake.params.pemmdb_path
    pemmdb = read_pemmdb(pemmdb_path)
    out = build_capacities(
        n, pemmdb, snakemake.params.pemmdb, snakemake.params.max_hours
    )

    pemmdb.to_csv(snakemake.output.pemmdb, index=False)
    with open(snakemake.output.capacities, "w") as f:
        f.write(
            f"# capacities from {pemmdb_path} for {snakemake.input.network}; "
            "capacity in p_nom/e_nom units (links: input at bus0), capacity_el in MW_el\n"
        )
        out.to_csv(f, index=False)
