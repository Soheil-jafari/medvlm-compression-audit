"""Pruning criteria. Each module registers one criterion by name.

| name                | module               | experiments            |
|---------------------|----------------------|------------------------|
| magnitude           | magnitude.py         | E5:magnitude           |
| activation          | activation.py        | E5:activation          |
| taylor              | taylor.py            | E5:taylor              |
| output_similarity   | output_similarity.py | E6                     |
| random              | random.py            | A3                     |
| clinical            | clinical.py          | E7, E8, A1, A2, A4-A6  |
"""

from __future__ import annotations

from medcompress.compression.pruning.criteria import (  # noqa: F401  (register criteria)
    activation,
    clinical,
    magnitude,
    output_similarity,
    random,
    taylor,
)
from medcompress.compression.pruning.criteria.base import CRITERIA, get_criterion

__all__ = ["CRITERIA", "get_criterion"]
