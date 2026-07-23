# PROPERGrid
![PROPERGrid Logo](LOGO_PROPER.png)

**Tools for Probabilistic Risk-Informed Operational Scheduling for Power Grids**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview
PROPERGrid is a powerful suite of tools designed to enhance operational scheduling for power grids using probabilistic risk-informed approaches. It enables grid operators and analysts to:
- Perform advanced risk assessments for grid operations
- Optimize scheduling decisions using probabilistic models
- Enhance grid reliability and efficiency through data-driven insights

## Key Features
- **Probabilistic Risk Assessment**
  - Monte Carlo simulation for uncertainty analysis
  - Fault tree and event tree analysis
  - Reliability metrics calculation
  
- **Operational Scheduling**
  - Risk-informed scheduling optimization
  - Multi-objective optimization support
  - Constraint handling for grid operations
  
- **Visualization and Reporting**
  - Interactive visualization of results
  - Customizable report generation
  - Real-time monitoring capabilities

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PROPERGrid.git
cd PROPERGrid
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from proper import Scheduler, RiskAssessment

# Initialize risk assessment
risk_model = RiskAssessment(
    confidence_level=0.95,
    time_horizon="24h"
)

# Configure scheduler
scheduler = Scheduler(
    grid_data="path/to/grid_data.csv",
    schedule_horizon=24,
    risk_model=risk_model
)

# Run optimization
result = scheduler.optimize()

# Export schedule_results
result.save_report("output/schedule_report.pdf")
```

## Documentation

- [User Guide](docs/source/user_guide/index.html) - Detailed usage instructions and examples
- [API Reference](docs/source/api/index.html) - Complete API documentation
- [Examples](docs/source/examples/index.html) - Example scripts and notebooks
- [Contributing](CONTRIBUTING.md) - Guidelines for contributing to PROPERGrid

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PROPERGrid in your research, please cite:
```bibtex
@software{propergrid2024,
  author = {Your Team},
  title = {PROPERGrid: Probabilistic Risk-Informed Operational Scheduling for Power Grids},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/PROPERGrid}
}
```
