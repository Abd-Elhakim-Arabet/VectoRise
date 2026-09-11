"""Data structures (VectorizeConfig, etc.)."""

from dataclasses import dataclass


@dataclass
class VectorizeConfig:
    """Configuration for a vectorization run."""

    input_path: str
    output_path: str
