import os
import json, json5
from pathlib import Path

import utils


def save_voice_presets_metadata(voice_presets_path, metadata):
    with open(voice_presets_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

def load_voice_presets_metadata(voice_presets_path, safe_if_metadata_not_exist=False):
    metadata_full_path = Path(voice_presets_path) / 'metadata.json'

    if safe_if_metadata_not_exist:
        if not os.path.exists(metadata_full_path):
            return {}

    with open(metadata_full_path, 'r') as f:
        presets = json5.load(f)

    return presets

# return system voice presets and session voice presets individually, each in a list
def get_voice_presets(session_id, lang):
    system_presets, session_presets = [], []

    # Load system presets
    system_presets = load_voice_presets_metadata(utils.get_system_voice_preset_path(lang))

    # Load session presets
    session_presets = load_voice_presets_metadata(
        utils.get_session_voice_preset_path(session_id),
        safe_if_metadata_not_exist=True
    )

    return system_presets, session_presets

# return merged voice presets in a {voice_preset_name: voice_preset} dict
def get_merged_voice_presets(session_id, lang):
    system_presets, session_presets = get_voice_presets(session_id, lang)
    res = {}
    for preset in list(system_presets.values()) + list(session_presets.values()):
        res[preset['id']] = preset  # session presets with the same id will cover that of system presets
    
    return res
