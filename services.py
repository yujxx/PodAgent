import os
import yaml
import logging
import nltk
import torch
import torchaudio
from torchaudio.transforms import SpeedPerturbation
from APIs import WRITE_AUDIO, LOUDNESS_NORM
from utils import fade, get_service_port
from flask import Flask, request, jsonify

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configure the logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a FileHandler for the log file
os.makedirs('services_logs', exist_ok=True)
log_filename = 'services_logs/Wav-API.log'
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the FileHandler to the root logger
logging.getLogger('').addHandler(file_handler)


"""
Initialize the AudioCraft models here
"""
from audiocraft.models import AudioGen, MusicGen
tta_model_size = config['AudioCraft']['tta_model_size']
tta_model = AudioGen.get_pretrained(f'facebook/audiogen-{tta_model_size}')
logging.info(f'AudioGen ({tta_model_size}) is loaded ...')

ttm_model_size = config['AudioCraft']['ttm_model_size']
ttm_model = MusicGen.get_pretrained(f'facebook/musicgen-{ttm_model_size}')
logging.info(f'MusicGen ({ttm_model_size}) is loaded ...')


"""
Initialize the CosyVoice2 here
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('TTS/CosyVoice')
sys.path.append('TTS/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
device = "cuda:1" if torch.cuda.is_available() else "cpu"
SPEED = float(config['Text-to-Speech']['speed'])
speed_perturb = SpeedPerturbation(32000, [SPEED])
cosyvoice = CosyVoice2('TTS/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
logging.info('CosyVoice2 model is loaded ...')

""" 
Initialize the VoiceFixer model here
"""
from voicefixer import VoiceFixer
vf = VoiceFixer()
logging.info('VoiceFixer is loaded ...')


app = Flask(__name__)


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    # Receive the text from the POST request
    data = request.json
    text = data['text']
    length = float(data.get('length', 5.0))
    volume = float(data.get('volume', -35))
    output_wav = data.get('output_wav', 'out.wav')

    logging.info(f'TTA (AudioGen): Prompt: {text}, length: {length} seconds, volume: {volume} dB')
    
    try:
        tta_model.set_generation_params(duration=length)  
        wav = tta_model.generate([text])  
        wav = torchaudio.functional.resample(wav, orig_freq=16000, new_freq=32000)

        wav = wav.squeeze().cpu().detach().numpy()
        wav = fade(LOUDNESS_NORM(wav, volumn=volume))
        WRITE_AUDIO(wav, name=output_wav)

        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Text-to-Audio generated successfully | {text}', 'file': output_wav})

    except Exception as e:
        return jsonify({'API error': str(e)}), 500


@app.route('/generate_music', methods=['POST'])
def generate_music():
    # Receive the text from the POST request
    data = request.json
    text = data['text']
    length = float(data.get('length', 5.0))
    volume = float(data.get('volume', -35))
    output_wav = data.get('output_wav', 'out.wav')

    logging.info(f'TTM (MusicGen): Prompt: {text}, length: {length} seconds, volume: {volume} dB')


    try:
        ttm_model.set_generation_params(duration=length)  
        wav = ttm_model.generate([text])  
        wav = wav[0][0].cpu().detach().numpy()
        wav = fade(LOUDNESS_NORM(wav, volumn=volume))
        WRITE_AUDIO(wav, name=output_wav)

        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Text-to-Music generated successfully | {text}', 'file': output_wav})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'API error': str(e)}), 500

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    # Receive the text from the POST request
    data = request.json
    text = data['text']
    speaker_id = data['speaker_id']
    ref_wav_path = data['speaker_path']
    instruct = data['speaking_style']
    volume = float(data.get('volume', -35))
    output_wav = data.get('output_wav', 'out.wav')
    SAMPLE_RATE = 24000
    
    logging.info(f'TTS (CosyVoice2): Speaker: {speaker_id}, Volume: {volume} dB, Prompt: {text}')

    try:   
        # Generate audio using the global pipe object
        text = text.replace('\n', ' ').strip()
        silence = torch.zeros(int(0.2 * SAMPLE_RATE)).unsqueeze(0) 
        spk_prompt = load_wav(ref_wav_path, 16000)

        pieces = []
        with torch.inference_mode():
            for i, j in enumerate(cosyvoice.inference_instruct2(text, instruct, spk_prompt, stream=False)):
                pieces += [j['tts_speech'], silence]

        result_audio = torch.cat(pieces, dim=1)
        wav_tensor = result_audio.to(dtype=torch.float32).cpu()
        wav = torchaudio.functional.resample(wav_tensor, orig_freq=SAMPLE_RATE, new_freq=32000)
        wav = speed_perturb(wav.float())[0].squeeze(0)
        wav = wav.numpy()
        wav = LOUDNESS_NORM(wav, volumn=volume)
        WRITE_AUDIO(wav, name=output_wav)

        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Text-to-Speech generated successfully | {speaker_id}: {text}', 'file': output_wav})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'API error': str(e)}), 500

@app.route('/fix_audio', methods=['POST'])
def fix_audio():
    # Receive the text from the POST request
    data = request.json
    processfile = data['processfile']

    logging.info(f'Fixing {processfile} ...')

    try:
        vf.restore(input=processfile, output=processfile, cuda=True, mode=0)
        
        # Return success message and the filename of the generated audio
        return jsonify({'message': 'Speech restored successfully', 'file': processfile})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'API error': str(e)}), 500


if __name__ == '__main__':
    service_port = get_service_port()
    # We disable multithreading to force services to process one request at a time and avoid CUDA OOM
    app.run(debug=False, threaded=False, port=service_port)
