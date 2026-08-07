"""Board-model lookup and Factory-namespace data layout for the factory scripts.

There is a provenance tag: "<value>,<provenance>".
Everything written by flash_factory_nominals.py is "nominal" (model-derived, not
measured).
"""

PROVENANCE_NOMINAL = "nominal"
KVS_VER = "1"

# EXC and AFE gain for v300-v500 are unverified guesses, as in the firmware script.
# Note that adc_gain is generated & read at runtime instead
BOARD_MODELS: dict[str, dict[str, str]] = {
    "v300": {"adc_fsr": "1.2", "exc": "4.53", "afe_gain": "50.0"},
    "v400": {"adc_fsr": "1.2", "exc": "4.53", "afe_gain": "50.0"},
    "v500": {"adc_fsr": "1.2", "exc": "4.53", "afe_gain": "50.0"},
    "v600L": {"adc_fsr": "1.2", "exc": "2.8", "afe_gain": "1.0"},
    "v600P": {"adc_fsr": "1.2", "exc": "4.53", "afe_gain": "101.0"},
    "v700L": {"adc_fsr": "1.2", "exc": "2.8", "afe_gain": "1.0"},
    "v700P": {"adc_fsr": "1.2", "exc": "4.53", "afe_gain": "101.0"},
}


def nominal_entries(board_model: str) -> dict[str, str]:
    """Factory-namespace keys for a board model: identity + provenance-tagged nominals."""
    entries = {"board_model": board_model, "kvs_ver": KVS_VER}
    for key, value in BOARD_MODELS[board_model].items():
        entries[key] = f"{value},{PROVENANCE_NOMINAL}"
    return entries
