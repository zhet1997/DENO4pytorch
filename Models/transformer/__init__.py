#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer models for neural operators
"""

from .Transformers import (
    SimpleTransformer,
    FourierTransformer,
    SimpleTransformerEncoderLayer,
    SimpleTransformerDecoderLayer,
    PointwiseRegressor,
    SpectralRegressor,
)

from .DualHeadTransformer import (
    CondLayerNorm,
    ConditionalTransformerEncoderLayer,
    DualHeadFourierTransformer,
)

__all__ = [
    # Original transformers
    'SimpleTransformer',
    'FourierTransformer',
    'SimpleTransformerEncoderLayer',
    'SimpleTransformerDecoderLayer',
    'PointwiseRegressor',
    'SpectralRegressor',
    # Dual-head conditional transformers
    'CondLayerNorm',
    'ConditionalTransformerEncoderLayer',
    'DualHeadFourierTransformer',
]

