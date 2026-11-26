# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Industry configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#industry_cf
"""

from typing import Literal

from pydantic import Field

from scripts.lib.validation.config._base import ConfigModel

StorageCarrier = Literal["battery", "H2"]


def _default_storage() -> list[StorageCarrier]:
    return ["battery", "H2"]


class _HourlyMatchingConfig(ConfigModel):
    """Configuration for `industry.hourly_matching` settings."""

    target: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Matching level of the `additional_flat_load`, e.g. 0.9 for 90%. 0 disables hourly matching (load supplied from the grid). Above 0, the load is supplied from dedicated, newly built VRE and storage subject to an hourly matching constraint added in `scripts/custom_extra_functionality.py`. See `mode` and `target_scope`.",
    )
    mode: Literal["hourly", "daily", "weekly", "monthly", "annual"] = Field(
        "hourly",
        description="Matching granularity: calendar periods (clock hours, days, weeks starting Monday, months, years) within which own VRE generation incl. storage and otherwise curtailed grid imports must match consumption. Within a period, excess exports offset grid imports. Snapshots are assigned to the period in which they start, so sub-hourly snapshots are summed per clock hour.",
    )
    target_scope: Literal["period", "annual"] = Field(
        "period",
        description="`period`: `target` must be met in every matching period. `annual`: `target` is the share of the annual consumption matched at the granularity of `mode`, i.e. unmatched energy can be distributed across periods (24/7 CFE score). Both are identical for `mode: annual` and for `target: 1`.",
    )
    carriers: list[str] = Field(
        default_factory=lambda: ["onwind", "solar"],
        description="Generator carriers the consumer may build. Costs and profiles are copied from the node; land potentials are shared with the grid generators.",
    )
    storage: list[StorageCarrier] = Field(
        default_factory=_default_storage,
        description="Storage technologies the consumer may build.",
    )
    fix_base: bool = Field(
        False,
        description="Only with `run: scenarios: enable`. Fix all capacities of every scenario (other than `base_scenario`) to the optimal capacities of the solved `base_scenario` of the same planning horizon, so that only the consumer's dedicated VRE (`carriers`) and battery can be built to meet the hourly matching at least cost. Dispatch is re-optimised. Other storage of the consumer (e.g. H2) is not built.",
    )
    base_scenario: str = Field(
        "base",
        description="Scenario whose solved capacities are fixed with `fix_base`.",
    )
    curtailed_import: bool = Field(
        True,
        description="Allow grid imports of electricity that would otherwise be curtailed (grid VRE of `curtailment_carriers` in the same country and hour) and count them as matched.",
    )
    curtailment_carriers: list[str] = Field(
        default_factory=lambda: [
            "onwind",
            "offwind-ac",
            "offwind-dc",
            "offwind-float",
            "solar",
            "solar-hsat",
            "solar rooftop",
        ],
        description="Grid generator carriers whose curtailment may be imported as matched electricity.",
    )
    max_excess_share: float | None = Field(
        None,
        ge=0.0,
        description="Upper bound on annual excess exports to the grid as share of the consumption (e.g. 0.2). `null` means unlimited.",
    )


class IndustryConfig(ConfigModel):
    """Configuration for `industry` settings."""

    St_primary_fraction: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0.6,
            2025: 0.55,
            2030: 0.5,
            2035: 0.45,
            2040: 0.4,
            2045: 0.35,
            2050: 0.3,
        },
        description="The fraction of steel produced via primary route versus secondary route (scrap+EAF). Current fraction is 0.6.",
    )
    DRI_fraction: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0,
            2025: 0,
            2030: 0.05,
            2035: 0.2,
            2040: 0.4,
            2045: 0.7,
            2050: 1,
        },
        description="The fraction of the primary route DRI + EAF.",
    )
    H2_DRI: float = Field(
        1.7,
        description="The hydrogen consumption in Direct Reduced Iron (DRI) Mwh_H2 LHV/ton_Steel from 51kgH2/tSt in `Vogl et al (2018) <https://doi.org/10.1016/j.jclepro.2018.08.279>`_.",
    )
    elec_DRI: float = Field(
        0.322,
        description="The electricity consumed in Direct Reduced Iron (DRI) shaft. From `HYBRIT brochure <https://web.archive.org/web/20210310021333/https://ssabwebsitecdn.azureedge.net/-/media/hybrit/files/hybrit_brochure.pdf>`_.",
    )
    Al_primary_fraction: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0.4,
            2025: 0.375,
            2030: 0.35,
            2035: 0.325,
            2040: 0.3,
            2045: 0.25,
            2050: 0.2,
        },
        description="The fraction of aluminium produced via the primary route versus scrap. Current fraction is 0.4.",
    )
    MWh_NH3_per_tNH3: float = Field(
        5.166,
        description="The energy amount per ton of ammonia (LHV).",
    )
    MWh_CH4_per_tNH3_SMR: float = Field(
        10.8,
        description="The energy amount of methane needed to produce a ton of ammonia using steam methane reforming (SMR). Value derived from 2012's demand from `Center for European Policy Studies (2008) <https://ec.europa.eu/docsroom/documents/4165/attachments/1/translations/en/renditions/pdf>`_.",
    )
    MWh_elec_per_tNH3_SMR: float = Field(
        0.7,
        description="The energy amount of electricity needed to produce a ton of ammonia using steam methane reforming (SMR). same source, assuming 94-6% split methane-elec of total energy demand 11.5 MWh/tNH3.",
    )
    MWh_H2_per_tNH3_electrolysis: float = Field(
        5.93,
        description="The energy amount of hydrogen needed to produce a ton of ammonia using Haber–Bosch process. From `Wang et al (2018) <https://doi.org/10.1016/j.joule.2018.04.017>`_, Base value assumed around 0.197 tH2/tHN3 (>3/17 since some H2 lost and used for energy).",
    )
    MWh_elec_per_tNH3_electrolysis: float = Field(
        0.2473,
        description="The energy amount of electricity needed to produce a ton of ammonia using Haber–Bosch process. From `Wang et al (2018) <https://doi.org/10.1016/j.joule.2018.04.017>`_, Table 13 (air separation and HB).",
    )
    MWh_NH3_per_MWh_H2_cracker: float = Field(
        1.46,
        description="The energy amount of amonia needed to produce an energy amount hydrogen using ammonia cracker.",
    )
    NH3_process_emissions: float = Field(
        24.5,
        description="The emission of ammonia production from steam methane reforming (SMR). From UNFCCC for 2015 for EU28.",
    )
    petrochemical_process_emissions: float = Field(
        25.5,
        description="The emission of petrochemical production. From UNFCCC for 2015 for EU28.",
    )
    HVC_primary_fraction: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0.88,
            2025: 0.85,
            2030: 0.78,
            2035: 0.7,
            2040: 0.6,
            2045: 0.5,
            2050: 0.4,
        },
        description="The fraction of high value chemicals (HVC) produced via primary route.",
    )
    HVC_mechanical_recycling_fraction: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0.12,
            2025: 0.15,
            2030: 0.18,
            2035: 0.21,
            2040: 0.24,
            2045: 0.27,
            2050: 0.30,
        },
        description="The fraction of high value chemicals (HVC) produced using mechanical recycling.",
    )
    HVC_chemical_recycling_fraction: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0.0,
            2025: 0.0,
            2030: 0.04,
            2035: 0.08,
            2040: 0.12,
            2045: 0.16,
            2050: 0.20,
        },
        description="The fraction of high value chemicals (HVC) produced using chemical recycling.",
    )
    HVC_environment_sequestration_fraction: float = Field(
        0.0,
        description="The fraction of high value chemicals (HVC) put into landfill resulting in additional carbon sequestration. The default value is 0.",
    )
    waste_to_energy: bool = Field(
        False,
        description="Switch to enable expansion of waste to energy CHPs for conversion of plastics. Default is false.",
    )
    waste_to_energy_cc: bool = Field(
        False,
        description="Switch to enable expansion of waste to energy CHPs for conversion of plastics with carbon capture. Default is false.",
    )
    solid_biomass_for_industry_cc: bool = Field(
        True,
        description="Add option for using solid biomass for industry with carbon capture, in addition to solid biomass for industry without carbon capture.",
    )
    gas_for_industry_cc: bool = Field(
        True,
        description="Add option for using gas for industry with carbon capture, in addition to gas for industry without carbon capture.",
    )
    process_emissions_cc: bool = Field(
        True,
        description="Add option for capturing industrial process emissions. If disabled, process emissions are always released to the atmosphere and can only be offset by negative emission technologies (e.g. direct air capture), which may make tight CO2 targets infeasible.",
    )
    sector_ratios_fraction_future: dict[int, float] = Field(
        default_factory=lambda: {
            2020: 0.0,
            2025: 0.05,
            2030: 0.2,
            2035: 0.45,
            2040: 0.7,
            2045: 0.85,
            2050: 1.0,
        },
        description="The fraction of total progress in fuel and process switching achieved in the industry sector.",
    )
    basic_chemicals_without_NH3_production_today: float = Field(
        69.0,
        description="The amount of basic chemicals produced without ammonia (= 86 Mtethylene-equiv - 17 MtNH3).",
    )
    HVC_production_today: float = Field(
        52.0,
        description="The amount of high value chemicals (HVC) produced. This includes ethylene, propylene and BTX. From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, Figure 16, page 107.",
    )
    MWh_elec_per_tHVC_mechanical_recycling: float = Field(
        0.547,
        description="The energy amount of electricity needed to produce a ton of high value chemical (HVC) using mechanical recycling. From SI of `Meys et al (2020) <https://doi.org/10.1016/j.resconrec.2020.105010>`_, Table S5, for HDPE, PP, PS, PET. LDPE would be 0.756.",
    )
    MWh_elec_per_tHVC_chemical_recycling: float = Field(
        6.9,
        description="The energy amount of electricity needed to produce a ton of high value chemical (HVC) using chemical recycling. The default value is based on pyrolysis and electric steam cracking. From `Material Economics (2019) <https://materialeconomics.com/latest-updates/industrial-transformation-2050>`_, page 125.",
    )
    chlorine_production_today: float = Field(
        9.58,
        description="The amount of chlorine produced. From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, Table 7, page 43.",
    )
    MWh_elec_per_tCl: float = Field(
        3.6,
        description="The energy amount of electricity needed to produce a ton of chlorine. From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, Table 6 page 43.",
    )
    MWh_H2_per_tCl: float = Field(
        -0.9372,
        description="The energy amount of hydrogen needed to produce a ton of chlorine. The value is negative since hydrogen produced in chloralkali process. From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, page 43.",
    )
    methanol_production_today: float = Field(
        1.5,
        description="The amount of methanol produced. From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, page 62.",
    )
    MWh_elec_per_tMeOH: float = Field(
        0.167,
        description="The energy amount of electricity needed to produce a ton of methanol from fossil gas. From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, Table 14, page 65.",
    )
    MWh_CH4_per_tMeOH: float = Field(
        10.25,
        description="The energy amount of methane needed to produce a ton of methanol from fossil gas. From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, Table 14, page 65.",
    )
    MWh_MeOH_per_tMeOH: float = Field(
        5.528,
        description="The energy amount per ton of methanol (LHV). From `DECHEMA (2017) <https://dechema.de/dechema_media/Downloads/Positionspapiere/Technology_study_Low_carbon_energy_and_feedstock_for_the_European_chemical_industry-p-20002750.pdf>`_, page 74.",
    )
    hotmaps_locate_missing: bool = Field(
        False,
        description="Locate industrial sites without valid locations based on city and countries.",
    )
    reference_year: int = Field(
        2023,
        description="The year used as the baseline for industrial energy demand and production. Data extracted from `JRC-IDEES 2015 <https://data.jrc.ec.europa.eu/dataset/jrc-10110-10001>`_.",
    )
    oil_refining_emissions: float = Field(
        0.013,
        description="The emissions from oil fuel processing (e.g. oil in petrochemical refinieries). The default value of 0.013 tCO2/MWh is based on DE statistics for 2019; the EU value is very similar.",
    )
    additional_flat_load: dict[str, float] = Field(
        default_factory=dict,
        description="Additional constant electricity demand in MW per country (e.g. `{DE: 100}`), added to the AC bus(es) of that country with carrier `industry electricity`.",
    )
    hourly_matching: _HourlyMatchingConfig = Field(
        default_factory=_HourlyMatchingConfig,
        description="Hourly matching of the `additional_flat_load`.",
    )
    temporal_electricity_industry_load: bool = Field(
        True,
        description="Use hourly load profiles per industry branch for the industrial electricity demand instead of a flat profile. The profiles from [FfE](https://opendata.ffe.de/dataset/normalized-industrial-electrical-load-profiles-germany/) are for Germany and are applied to all countries.",
    )
