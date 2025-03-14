import os
import re
import torch
import  numpy as np
import yaml
import json5
from pathlib import Path


#### path related code BEGIN ####
def get_session_path(session_id):
    return Path(f'output/sessions/{session_id}')

def get_system_voice_preset_path(lang):
    if lang =='zh':
        return Path('data/voice_presets_cv_zh')
    else:
        return Path('data/voice_presets_cv_en')

def get_session_voice_preset_path(session_id):
    return Path(f'{get_session_path(session_id)}/voice_presets')
    
def get_session_audio_path(session_id):
    return Path(f'{get_session_path(session_id)}/audio')

def rescale_to_match_energy(segment1, segment2):
    ratio = get_energy_ratio(segment1, segment2)
    recaled_segment1 = segment1 / ratio
    return recaled_segment1.numpy()
#### path related code END ####

def text_to_abbrev_prompt(input_text):
    return re.sub(r'[^a-zA-Z_]', '', '_'.join(input_text.split()[:5]))

def get_energy(x):
    return np.mean(x ** 2)


def get_energy_ratio(segment1, segment2):
    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.tensor(ratio)
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio

def fade(audio_data, fade_duration=2, sr=32000):
    audio_duration = audio_data.shape[0] / sr

    # automated choose fade duration
    if audio_duration >=8:
         # keep fade_duration 2
        pass
    else:
        fade_duration = audio_duration / 5

    fade_sampels = int(sr * fade_duration)
    fade_in = np.linspace(0, 1, fade_sampels)
    fade_out = np.linspace(1, 0, fade_sampels)

    audio_data_fade_in = audio_data[:fade_sampels] * fade_in
    audio_data_fade_out = audio_data[-fade_sampels:] * fade_out

    audio_data_faded = np.concatenate((audio_data_fade_in, audio_data[len(fade_in):-len(fade_out)], audio_data_fade_out))
    return audio_data_faded

def get_service_port():
    service_port = os.environ.get('PODAGENT_SERVICE_PORT')
    return service_port

def get_service_url():
    service_url = os.environ.get('PODAGENT_SERVICE_URL')
    return service_url 

def get_api_key():
    api_key = os.environ.get('PODAGENT_OPENAI_KEY')
    return api_key       

def get_max_script_lines():
    max_lines = int(os.environ.get('PODAGENT_MAX_SCRIPT_LINES', 999))
    return max_lines

def check_conversation_script(data):
    for line in data:
        if isinstance(line, dict):
            line_json = line
        else:
            line_json = json5.loads(line)
        check = line_json['speaker']
        check = line_json['speaking_content']
        check = line_json['speaking_style']

def complete_speech_content(audio_script_response, conversation_script):
    complte_audio_script = []
    for audio_item in audio_script_response:
        if audio_item['audio_type'] == 'speech':
            speech_item = conversation_script[audio_item['index']]
            complete_speech_item = {
                "audio_type": "speech",
                "layout": "foreground",
                "character": speech_item["speaker"],
                "vol": -15,
                "text": speech_item["speaking_content"],
                "speaking_style": speech_item["speaking_style"]
            }
            complte_audio_script.append(complete_speech_item)
        else:
            complte_audio_script.append(audio_item)
    
    return complte_audio_script

def check_by_audio_type(audio, mandatory_attrs_map, audio_str):
        if audio['audio_type'] not in mandatory_attrs_map:
            raise ValueError('audio_type is not allowed in this layout, audio={audio_str}')
        for attr_name in mandatory_attrs_map[audio['audio_type']]:
            if attr_name not in audio:
                raise ValueError(f'{attr_name} does not exist, audio={audio_str}')
            
def check_json_script(data, conversation_script):
    # check speech item index
    indice_num = 0
    expected_indice = -1
    for item in data:
        if item['audio_type'] == 'speech':
            indice_num += 1
            if expected_indice == -1:
                expected_indice = item['index'] + 1
                continue
            if item['index'] == expected_indice:
                expected_indice += 1
            else:
                raise ValueError(f"Speech index is not in order.")
    
    if indice_num > len(conversation_script):
        raise ValueError(f"Speech item is out of index.")
    if indice_num < len(conversation_script):
        raise ValueError(f"Speech item is less than provided.")

    # complete speech content if no index issue
    data = complete_speech_content(data, conversation_script)
    
    # check all audio items
    foreground_mandatory_attrs_map = {
        'music': ['vol', 'len', 'desc'],
        'sound_effect': ['vol', 'len', 'desc'],
        'speech': ['vol', 'text']
    }
    background_mandatory_attrs_map = {
        'music': ['vol', 'desc'],
        'sound_effect': ['vol', 'desc'],
    }

    # Check json's format
    for audio in data:
        audio_str = json5.dumps(audio, indent=None)
        if 'layout' not in audio:
            raise ValueError(f'layout missing, audio={audio_str}')
        elif 'audio_type' not in audio:
            raise ValueError(f'audio_type missing, audio={audio_str}')
        elif audio['layout'] == 'foreground':
            check_by_audio_type(audio, foreground_mandatory_attrs_map, audio_str)
        elif audio['layout'] == 'background':
            if 'id' not in audio:
                raise ValueError(f'id not in background audio, audio={audio_str}')
            if 'action' not in audio:
                raise ValueError(f'action not in background audio, audio={audio_str}')
            if audio['action'] == 'begin':
                check_by_audio_type(audio, background_mandatory_attrs_map, audio_str)
            else:
                if audio['action'] != 'end':
                    raise ValueError(f'Unknown action, audio={audio_str}')
        else:
            raise ValueError(f'Unknown layout, audio={audio_str}')
    
    return data

def convert_bg_to_fg(bg_audio):
    fg_audio = {k: v for k, v in bg_audio.items() if k not in ['action', 'begin_fg_audio_id', 'end_fg_audio_id']}
    fg_audio['layout'] = 'foreground'  
    fg_audio['id'] = bg_audio['begin_fg_audio_id']

    # Add 'len' based on 'audio_type'
    if fg_audio['audio_type'] == 'music':
        fg_audio['len'] = 10
    elif fg_audio['audio_type'] == 'sound_effect':
        fg_audio['len'] = 2

    return fg_audio

def load_name_list(bg_info):
    name_list = []

    for key in bg_info:
        value = bg_info[key]
        
        if isinstance(value, list) and len(value) == 1:
            value = value[0]

        if 'Name' in value:
            name_list.append(value["Name"])
        elif 'name' in value:
            name_list.append(value["name"])
    
    return name_list

def check_map(char_voice_map, bg_info):
    std_name_list = load_name_list(bg_info)

    name_dict = {name.replace('_', '').replace(' ', '').lower(): name for name in std_name_list}
    name_dict['host'] = 'Host'
    
    for key in char_voice_map:
        name_key = key.replace('_', '').replace(' ', '').lower()
        if name_key not in name_dict:
            print(f'{name_key} not in:')
            print(name_dict)
        if key != name_dict[name_key]:
            char_voice_map[name_dict[name_key]] = char_voice_map.pop(key)
    
    return char_voice_map
            