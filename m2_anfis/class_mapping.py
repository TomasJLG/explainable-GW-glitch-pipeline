"""
class_mapping.py
Maps 23 Gravity Spy fine-grained labels to 5 macro-classes.
"""

FINE_TO_MACRO: dict[str, str] = {
    # Loud transients
    "Extremely_Loud":      "Loud",
    "Chirp":               "Loud",
    "Koi_Fish":            "Loud",
    # Burst / broadband
    "Blip":                "Burst",
    "Blip_Low_Frequency":  "Burst",
    "Repeating_Blips":     "Burst",
    "Low_Frequency_Burst": "Burst",
    "Fast_Scattering":     "Burst",
    "Scratchy":            "Burst",
    # Scattered-light arches
    "Scattered_Light":     "Scatter",
    "Wandering_Line":      "Scatter",
    # Spectral lines
    "Low_Frequency_Lines": "Line",
    "60Hz_Line":           "Line",
    "Whistle":             "Line",
    "Violin_Mode":         "Line",
    "Violin_Mode_Harmonic":"Line",
    "Power_Line":          "Line",
    # Other / unclassified
    "No_Glitch":           "Other",
    "Air_Compressor":      "Other",
    "Helix":               "Other",
    "Light_Modulation":    "Other",
    "Paired_Doves":        "Other",
    "Tomte":               "Other",
}

MACRO_NAMES: list[str] = ["Loud", "Burst", "Scatter", "Line", "Other"]
MACRO_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(MACRO_NAMES)}


def to_macro(fine_label: str) -> str:
    """Return macro-class name; falls back to 'Other' for unknown labels."""
    return FINE_TO_MACRO.get(fine_label, "Other")


def to_idx(fine_label: str) -> int:
    return MACRO_TO_IDX[to_macro(fine_label)]
