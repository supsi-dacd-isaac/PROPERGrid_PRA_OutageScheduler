# PROPERGrid
![PROPERGrid Logo](LOGO_PROPER.pdf)

**Tools for Probabilistic Risk-Informed Operational Scheduling for Power Grids**

## Overview
PROPERGrid is a powerful suite of tools designed to enhance operational scheduling for power grids using probabilistic risk-informed approaches. It allows users to integrate advanced risk assessments and probabilistic models to optimize grid operations and scheduling, improving decision-making and efficiency.

## Features
- **Probabilistic Risk Assessment**: Integrates probabilistic modelling to assess and manage operational risks.
- **Operational Scheduling Problems**: Optimizes grid operation schedules based on risk-informed analysis.
 

## Installation

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage Example

Here's a basic usage example for performing a risk-informed operational schedule:

```python
from propergrid import Scheduler, RiskAssessment

# Initialize scheduler with desired settings
scheduler = Scheduler(grid_data="path/to/grid_data", schedule_horizon=24)

# Perform risk-informed operational scheduling
result = scheduler.run()

# Display or save results
print(result)
result.save("path/to/output_schedule.json")
```

## Configuration

- `grid_data`: Path to the grid data file (in CSV, JSON, or other supported formats).
- `schedule_horizon`: The scheduling horizon (number of hours or days) for optimization.

## Contributions

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
 
