"""Production defaults and explicit registrations for historical comparisons.

The JSON registrations remain immutable experiment records. Change the default
here rather than overwriting a historical registration with a different system.
"""

LEGACY_REGISTRATION = 'configs/controller/current_baseline.json'
B_REGISTRATION = 'configs/controller/quality_baseline_v1.json'
BM25_REGISTRATION = 'configs/controller/quality_bm25_v1.json'
DEFAULT_REGISTRATION = 'configs/controller/quality_adaptive_v1.json'

QUALITY_REGISTRATIONS = (B_REGISTRATION, BM25_REGISTRATION, DEFAULT_REGISTRATION)
REGISTRATIONS = (LEGACY_REGISTRATION, *QUALITY_REGISTRATIONS)
