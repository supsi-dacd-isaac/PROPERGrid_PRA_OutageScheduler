# PROPERGrid

![PROPERGrid logo](LOGO_PROPER.png)

**Probabilistic risk assessment, security analysis, and risk-informed outage scheduling for transmission power grids.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Project](https://img.shields.io/badge/SFOE-PROPER--Grids-orange)](https://www.aramis.admin.ch/Grunddaten/?ProjectID=53864&Sprache=en-US)

## Overview

PROPERGrid is a research software repository developed within the **PROPER-Grids — Probabilistic Risk-informed Operational Scheduling for Power Grids** project.

The repository brings together tools for:

- loading and preparing transmission-system test cases;
- probabilistic risk assessment (PRA) and probabilistic security assessment (PSA);
- N−1 and N−k contingency generation and analysis;
- AC and DC power-flow simulation;
- LODF-based contingency screening;
- severity and risk aggregation;
- Monte Carlo and rare-event sampling experiments;
- deterministic and CVaR-based transmission-outage scheduling;
- clustered decomposition methods for larger scheduling studies;
- visualisation of networks, outage schedules, contingency results, and scheduler comparisons.

The code is intended for research, benchmarking, teaching, and method development. It is not a production operational-planning platform and does not replace the security-analysis procedures of a transmission system operator.

## Repository scope

The repository contains four main methodological areas.

### 1. Probabilistic risk and security assessment

The `pra_psa` package provides the main PRA/PSA functionality:

- system-data loading from MATPOWER-style files;
- N−1 and N−k contingency construction;
- AC and DC contingency analysis;
- line-outage distribution factor screening;
- load-flow, contingency, and simplified cascade simulations;
- severity metrics;
- contingency-probability models;
- risk aggregation by contingency and operating period;
- risk curves, Value-at-Risk, and Conditional Value-at-Risk;
- Monte Carlo and experimental subset-simulation utilities.

A runnable IEEE RTS-24 example is provided in:

```text
demos/run_ieee24_contingency_analysis.py
```

### 2. Probabilistic operating-state models

The `probabilistic_models` package contains lightweight models for sampling uncertain demand and defining probabilistic inputs to PRA/PSA and scheduling studies.

Additional notebooks in `demos` illustrate efficient sampling, importance sampling, subset simulation, and CVaR-oriented experiments.

### 3. Outage scheduling and optimisation

The repository contains several related scheduling implementations:

- `scheduler`: reference deterministic security-constrained outage-scheduling formulation;
- `optimizers`: monolithic deterministic and CVaR scheduling models, supporting utilities, post-processing, and comparison scripts;
- `scheduler_clustered`: deterministic and CVaR-guided outage-cluster decomposition methods designed to reduce repeated operational-security evaluations.

The clustered implementation groups consecutive periods having the same planned-outage topology and evaluates them through security-constrained operational subproblems. Detailed mathematical and implementation notes are kept in:

- [`scheduler_clustered/README_CLUSTERED_DETERMINISTIC.md`](scheduler_clustered_v0/README_CLUSTERED_DETERMINISTIC.md)
- [`scheduler_clustered/README_CLUSTERED_CVAR.md`](scheduler_clustered_v0/README_CLUSTERED_CVAR.md)

### 4. Visualisation and demonstrations

The `visualization` package includes tools for:

- outage-schedule plots;
- deterministic and CVaR result comparisons;
- clustered-scheduler diagnostics;
- IEEE and Swiss transmission-system maps.

The `demos` directory contains scripts and notebooks for the principal use cases.

## Repository structure

```text
PROPERGrid_PRA_OutageScheduler/
├── config/                  # Case and scheduling configuration files
├── data/
│   └── powersystems/        # Power-system cases and processed demand data
├── demos/                   # Runnable examples and research notebooks
├── optimizers/              # Monolithic deterministic and CVaR schedulers
├── outputs/                 # Example and generated results
├── pra_psa/                 # PRA/PSA, contingency analysis, risk and simulation
│   ├── core/
│   ├── sampler/
│   └── simulation/
├── probabilistic_models/    # Probabilistic demand and uncertainty models
├── scheduler/               # Reference deterministic outage scheduler
├── scheduler_clustered/     # Clustered deterministic and CVaR schedulers
├── tests/                   # Automated tests
├── utils/                   # Shared preprocessing and utility functions
├── visualization/           # Network and scheduling visualisation
├── run_clustered_comparison.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

### Requirements

- Python 3.10 or later;
- `pip`;
- dependencies listed in `requirements.txt`;
- a valid Gurobi installation and licence for the optimisation and scheduling modules.

The PRA/PSA and data-analysis components can be used without Gurobi where the selected example does not invoke an optimisation model.

### Clone and install

```bash
git clone https://github.com/supsi-dacd-isaac/PROPERGrid_PRA_OutageScheduler.git
cd PROPERGrid_PRA_OutageScheduler

python -m venv .venv
```

Activate the environment.

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the repository dependencies and the PRA/PSA package:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For scheduling experiments, verify the Gurobi installation:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## Quick start

Run all commands from the repository root.

### IEEE RTS-24 contingency analysis

```bash
python demos/run_PRA_contingency_analysis.py
```

This example:

1. loads the IEEE RTS-24 network and demand data;
2. constructs line, transformer, and generator N−1 contingencies;
3. performs LODF screening where applicable;
4. runs contingency analysis over selected operating points;
5. assigns contingency probabilities;
6. produces contingency-risk tables and risk curves.

Results are written to:

```text
outputs/ieee24_contingency/
```

### Reference deterministic outage scheduler

```bash
python -m scheduler.main
```

The default configuration is selected inside `scheduler/main.py`.

### Clustered deterministic scheduler

```bash
python -m scheduler_clustered_v0.main_clustered
```

### Clustered CVaR-guided scheduler

```bash
python -m scheduler_clustered_v0.main_clustered_cvar
```

### Compare clustered deterministic and CVaR results

```bash
python run_clustered_comparison.py
```

To regenerate plots from existing result files:

```bash
python run_clustered_comparison.py --reuse-results
```

Comparison runs are written under:

```text
outputs/clustered_comparison_runs/
```

### Run tests

```bash
pytest
```

## Configuration

Case-specific settings are stored in `config/`. Current examples include configurations for:

- IEEE RTS-24;
- IEEE 118-bus;
- the Swiss transmission-system research case.

Configuration files define parameters such as:

- case and data paths;
- pandapower case selection;
- temporal aggregation;
- maintained assets;
- maintenance costs, durations, and priorities;
- load scaling;
- value of lost load;
- scenario counts;
- limits on simultaneous maintenance activities.

The executable scheduling scripts currently select their configuration files in code. Check the `conf_path` argument in the relevant entry point before running a new case.

## Data organisation

The data loaders support MATPOWER-style network files and processed demand time series. A typical case directory is:

```text
data/powersystems/<system_name>/
├── System.m or System.xlsx
└── hourlyDemandBus.pkl
```

The exact files required depend on the selected module and configuration.

Before running a new case, verify that:

- bus identifiers and demand columns are aligned;
- generator and branch identifiers are consistent;
- branch ratings and generator limits are meaningful;
- maintained-component identifiers exist in the network;
- the base case is solvable;
- contingency definitions remain valid after conversion.

## Data availability

### IEEE test systems

Selected IEEE test-case files and processed examples are included where redistribution is permitted.

Additional or complete datasets used in the IEEE RTS-24 and IEEE 118-bus studies are available from the project team **upon reasonable request**, subject to the licences and redistribution conditions of the original data sources.

Requests should specify:

- the requested test system;
- the intended research use;
- the requesting institution;
- a contact person.

Data requests may be submitted through a GitHub issue or to:

**Roberto Rocchetta**  
SUPSI, DACD–ISAAC  
`roberto.rocchetta@supsi.ch`

### Swissgrid system

The Swiss transmission-system model and the associated operational, demand, generation, outage, and disturbance data are protected by confidentiality obligations and a non-disclosure agreement with **Swissgrid**.

These datasets are not distributed through this public repository. Access cannot be granted solely by the repository maintainers and is subject to formal authorisation by the data owner and the relevant project partners.

The public repository contains case-independent methods, interfaces, and selected visualisation tools that can be applied to an authorised Swissgrid dataset in an approved environment.

## Outputs and reproducibility

Depending on the selected workflow, the repository produces:

- contingency-analysis tables;
- risk by contingency and operating period;
- risk and exceedance curves;
- outage schedules;
- DNS and load-curtailment indicators;
- deterministic and CVaR optimisation summaries;
- JSON, CSV, pickle, and figure outputs;
- run manifests and logs.

For reproducible research, report at least:

- the repository commit;
- configuration file;
- test system and data version;
- contingency set;
- power-flow model;
- uncertainty and sampling settings;
- random seeds;
- solver version and parameters;
- hardware and parallelisation settings.

## Current status and limitations

PROPERGrid is an active research repository. Some modules are mature experimental implementations, while others remain exploratory or provide compatibility with earlier project developments.

Important limitations include:

- multiple scheduling implementations coexist and do not expose one unified stable API;
- some notebooks and scripts are research demonstrations rather than maintained command-line applications;
- optimisation modules require case-specific calibration and a Gurobi licence;
- the clustered schedulers are decomposition-based research methods and do not provide a global optimality certificate for the complete stochastic scheduling problem;
- DC models are used in several scheduling formulations, while AC analysis is available in the PRA/PSA workflow;
- uncertainty models and contingency probabilities must be calibrated for each application;
- Swissgrid studies require restricted data that are not part of the public repository.

Users should inspect the configuration and the relevant module-level documentation before interpreting numerical results.

## Funding and acknowledgements

This repository was developed within:

> **PROPER-Grids — Probabilistic Risk-informed Operational Scheduling for Power Grids**

The project is funded by the **Swiss Federal Office of Energy (SFOE)** under contract:

> **SI/502663-01**

Official project information:

https://www.aramis.admin.ch/Grunddaten/?ProjectID=53864&Sprache=en-US

The project is led by **SUPSI** in collaboration with **ETH Zürich**, with **Swissgrid** participating as an industry stakeholder and data partner under the applicable confidentiality arrangements.

## Licence

The source code is distributed under the **Apache License 2.0**. See [LICENSE](LICENSE).

Third-party software, network models, and datasets remain subject to their own licences and terms of use. The Apache-2.0 licence of this repository does not grant rights to redistribute external or NDA-protected data.

## Citation

A versioned software citation and DOI will be added for an archived release. Until then, cite the repository as:

```bibtex
@software{propergrid2026,
  author       = {Rocchetta, Roberto and Nespoli, Lorenzo and Gjorgiev, Blazhe
                  and Sansavini, Giovanni and Medici, Vasco},
  title        = {PROPERGrid: Probabilistic Risk Assessment and
                  Risk-Informed Outage Scheduling for Power Grids},
  year         = {2026},
  publisher    = {GitHub},
  organization = {SUPSI and ETH Zurich},
  url          = {https://github.com/supsi-dacd-isaac/PROPERGrid_PRA_OutageScheduler}
}
```

When publishing results obtained with this repository, also cite the relevant methodological papers and original test-system data sources.

## Contact

**Roberto Rocchetta**  
SUPSI - Institute for Applied Sustainability to the Built Environment Department of Environment, Constructions and Design,  
`roberto.rocchetta@supsi.ch`
