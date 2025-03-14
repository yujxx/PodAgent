import os
import json5
import utils

def collect_and_check_audio_data(data):
    fg_audio_id = 0
    fg_audios = []
    bg_audios = []
    # Collect all the foreground and background audio ids used to calculate background audio length later
    for audio in data:
        if audio['layout'] == 'foreground':
            audio['id'] = fg_audio_id
            fg_audios.append(audio)
            fg_audio_id += 1
        else:   # background
            if audio['action'] == 'begin':
                audio['begin_fg_audio_id'] = fg_audio_id
                bg_audios.append(audio)
            else:   # ends
                # find the backgound with the id, and update its 'end_fg_audio_id'
                for bg_audio in bg_audios:
                    if bg_audio['id'] == audio['id'] and bg_audio['audio_type'] == audio['audio_type']:
                        bg_audio['end_fg_audio_id'] = fg_audio_id
                        break
    
    # check if all background audios are valid
    for bg_audio in bg_audios:
        if 'begin_fg_audio_id' not in bg_audio:
            raise ValueError(f'begin of background missing, audio={bg_audio}')
        elif 'end_fg_audio_id' not in bg_audio:
            raise ValueError(f'end of background missing, audio={bg_audio}')

        if bg_audio['begin_fg_audio_id'] > bg_audio['end_fg_audio_id']:
            raise ValueError(f'background audio ends before start, audio={bg_audio}')
        elif bg_audio['begin_fg_audio_id'] == bg_audio['end_fg_audio_id']:
            raise ValueError(f'background audio contains no foreground audio, audio={bg_audio}')
            #fg_audio = convert_bg_to_fg(bg_audio)
            #fg_audios.append(fg_audio)
            #bg_audios.remove(bg_audio)
            
        #except Exception as err:
        #    sys.stderr.write(f'ALIGNMENT ERROR: {err}, audio={bg_audio}\n')
        #    return None, None

    return fg_audios, bg_audios


class AudioCodeGenerator:
    def __init__(self):
        self.wav_counters = {
            'bg_sound_effect': 0,
            'bg_music': 0,
            'idle': 0,
            'fg_sound_effect': 0,
            'fg_music': 0,
            'fg_speech': 0,
        }
        self.code = ''
    
    def append_code(self, content):
        self.code = f'{self.code}{content}\n'

    def generate_code(self, fg_audios, bg_audios, output_path, result_filename):
        def get_wav_name(audio):
            audio_type = audio['audio_type']
            layout = 'fg' if audio['layout'] == 'foreground' else 'bg'
            wav_type = f'{layout}_{audio_type}' if layout else audio_type
            desc = audio['text'] if 'text' in audio else audio['desc']
            desc = utils.text_to_abbrev_prompt(desc)
            wav_filename = f'{wav_type}_{self.wav_counters[wav_type]}_{desc}.wav'
            self.wav_counters[wav_type] += 1
            return wav_filename

        header = f'''
import os
import sys
import datetime
import time

from APIs import TTM, TTS, TTA, MIX, CAT, COMPUTE_LEN


fg_audio_lens = []
wav_path = \"{output_path.absolute()}/audio\"
os.makedirs(wav_path, exist_ok=True)

'''
        self.append_code(header)

        fg_audio_wavs = []
        for fg_audio in fg_audios:
            wav_name = get_wav_name(fg_audio)
            if fg_audio['audio_type'] == 'sound_effect':
                self.append_code(f'TTA(text=\"{fg_audio["desc"]}\", length={fg_audio["len"]}, volume={fg_audio["vol"]}, out_wav=os.path.join(wav_path, \"{wav_name}\"))')
            elif fg_audio['audio_type'] == 'music':
                self.append_code(f'TTM(text=\"{fg_audio["desc"]}\", length={fg_audio["len"]}, volume={fg_audio["vol"]}, out_wav=os.path.join(wav_path, \"{wav_name}\"))')
            elif fg_audio['audio_type'] == 'speech':
                ref_path = ""
                if "npz_path" in self.char_to_voice_map[fg_audio["character"]]:
                    ref_path = self.char_to_voice_map[fg_audio["character"]]["npz_path"]
                if "wav_path" in self.char_to_voice_map[fg_audio["character"]]:
                    ref_path = self.char_to_voice_map[fg_audio["character"]]["wav_path"]
                ref_full_path = os.path.abspath(ref_path) if os.path.exists(ref_path) else ref_path

                self.append_code(f'TTS(text=\"{fg_audio["text"]}\", speaker_id=\"{self.char_to_voice_map[fg_audio["character"]]["id"]}\", volume={fg_audio["vol"]}, out_wav=os.path.join(wav_path, \"{wav_name}\"), speaker_path=\"{ref_full_path}\", speaking_style=\"{fg_audio["speaking_style"]}\")')

            fg_audio_wavs.append(wav_name)
            self.append_code(f'fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, \"{wav_name}\")))\n')
        
        # cat all foreground audio together
        self.append_code(f'fg_audio_wavs = []')
        for wav_filename in fg_audio_wavs:
            self.append_code(f'fg_audio_wavs.append(os.path.join(wav_path, \"{wav_filename}\"))')
        self.append_code(f'CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, \"foreground.wav\"))')

        bg_audio_wavs = []
        self.append_code(f'\nbg_audio_offsets = []')
        for bg_audio in bg_audios:
            wav_name = get_wav_name(bg_audio)
            self.append_code(f'bg_audio_len = sum(fg_audio_lens[{bg_audio["begin_fg_audio_id"]}:{bg_audio["end_fg_audio_id"]}])')
            self.append_code(f'bg_audio_offset = sum(fg_audio_lens[:{bg_audio["begin_fg_audio_id"]}])')
            if bg_audio['audio_type'] == 'sound_effect':
                self.append_code(f'TTA(text=\"{bg_audio["desc"]}\", volume={bg_audio["vol"]}, length=bg_audio_len, out_wav=os.path.join(wav_path, \"{wav_name}\"))')
            elif bg_audio['audio_type'] == 'music':
                self.append_code(f'TTM(text=\"{bg_audio["desc"]}\", volume={bg_audio["vol"]}, length=bg_audio_len, out_wav=os.path.join(wav_path, \"{wav_name}\"))')
            else:
                raise ValueError()
            bg_audio_wavs.append(wav_name)
            self.append_code(f'bg_audio_offsets.append(bg_audio_offset)\n')
        self.append_code(f'bg_audio_wavs = []')
        for wav_filename in bg_audio_wavs:
            self.append_code(f'bg_audio_wavs.append(os.path.join(wav_path, \"{wav_filename}\"))')

        self.append_code(f'bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))')
        self.append_code(f'bg_audio_wav_offset_pairs.append((os.path.join(wav_path, \"foreground.wav\"), 0))')
        self.append_code(f'MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, \"{result_filename}.wav\"))')

    def init_char_to_voice_map(self, filename):
        with open(filename, 'r') as file:
            self.char_to_voice_map = json5.load(file)


    def parse_and_generate(self, script_filename, char_to_voice_map_filename, output_path, result_filename='result'):
        self.code = ''
        self.init_char_to_voice_map(char_to_voice_map_filename)

        data = []
        with open(script_filename, 'r') as file:
            for line in file:
                line = line.strip()  # 去除行首尾空白
                if line:  # 确保行非空
                    json_object = json5.loads(line)
                    data.append(json_object)

        fg_audios, bg_audios = collect_and_check_audio_data(data)
        self.generate_code(fg_audios, bg_audios, output_path, result_filename)
        return self.code
