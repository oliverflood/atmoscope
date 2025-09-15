from dataclasses import dataclass

@dataclass
class LabelSpace:
    name: str
    classes: list[str]

COARSE_7 = LabelSpace("coarse7", [
    "altostratus_stratus", 
    "cirrocumulus_altocumulus",
    "cirrus_cirrostratus", 
    "clearsky",
    "cumulonimbus", 
    "cumulus", 
    "stratocumulus"
])

ICA_10 = LabelSpace("ica10", [
    "altocumulus",
    "altostratus",
    "cirrocumulus",
    "cirrostratus",
    "cirrus",
    "cumulonimbus",
    "cumulus",
    "nimbostratus",
    "stratocumulus",
    "stratus"
])