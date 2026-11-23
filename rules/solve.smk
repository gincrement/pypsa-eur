# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule build_pemmdb_capacities:
    """Maps PEMMDB capacities per market node and technology onto network components."""
    input:
        network=resources("networks/composed_{horizon}.nc"),
        pemmdb=input_pemmdb,
    output:
        capacities=resources("fixed_capacities_pemmdb_{horizon}.csv"),
        pemmdb=resources("pemmdb_{horizon}.csv"),
    log:
        logs("build_pemmdb_capacities_{horizon}.log"),
    params:
        pemmdb=config_provider("solving", "options", "fixed_capacities", "pemmdb"),
        pemmdb_path=pemmdb_path,
        max_hours=config_provider("electricity", "max_hours"),
    resources:
        mem_mb=4000,
    script:
        scripts("build_pemmdb_capacities.py")


rule solve_network:
    """Optimises investment and dispatch for one planning horizon to minimise total system cost."""
    input:
        network=resources("networks/composed_{horizon}.nc"),
        base_network=input_fixed_base_network,
        fixed_capacities=input_fixed_capacities,
    output:
        network=RESULTS + "networks/solved_{horizon}.nc",
        model=(
            RESULTS + "models/solved_{horizon}.nc"
            if config["solving"]["options"]["store_model"]
            else []
        ),
    log:
        solver=normpath(RESULTS + "logs/solve_network/solver_{horizon}.log"),
        memory=RESULTS + "logs/solve_network/memory_{horizon}.log",
        python=RESULTS + "logs/solve_network/python_{horizon}.log",
    benchmark:
        (RESULTS + "benchmarks/solve_network_{horizon}.log")
    shadow:
        shadow_config
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("planning_horizons"),
        sector=config_provider("sector"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential"
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    script:
        scripts("solve_network.py")


rule solve_operations_network:
    """Re-optimises dispatch only, holding the capacities from the solved network fixed."""
    input:
        network=RESULTS + "networks/solved_{horizon}.nc",
    output:
        network=RESULTS + "networks/operations_{horizon}.nc",
    log:
        solver=normpath(RESULTS + "logs/solve_operations_network/solver_{horizon}.log"),
        memory=RESULTS + "logs/solve_operations_network/memory_{horizon}.log",
        python=RESULTS + "logs/solve_operations_network/python_{horizon}.log",
    benchmark:
        (RESULTS + "benchmarks/solve_operations_network_{horizon}.log")
    shadow:
        shadow_config
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("planning_horizons"),
        sector=config_provider("sector"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential"
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    script:
        scripts("solve_operations_network.py")
