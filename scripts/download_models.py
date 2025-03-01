import yaml
import os

# Read the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract values for each application
ttm_model_size = config['AudioCraft']['ttm_model_size']
tta_model_size = config['AudioCraft']['tta_model_size']

# Download nltk
import nltk
nltk.download('punkt')

print('Step 1: Downloading TTS model ...')
from modelscope import snapshot_download
os.makedirs('TTS/CosyVoice/pretrained_models', exist_ok=True)
snapshot_download("iic/CosyVoice2-0.5B", local_dir="TTS/CosyVoice/pretrained_models/CosyVoice2-0.5B")

print('Step 2: Downloading TTA model ...')
from audiocraft.models import AudioGen
tta_model = AudioGen.get_pretrained(f"facebook/audiogen-{tta_model_size}")

print('Step 3: Downloading TTM model ...')
from audiocraft.models import MusicGen
tta_model = MusicGen.get_pretrained(f"facebook/musicgen-{ttm_model_size}")

print('Step 4: Downloading SR model ...')
from voicefixer import VoiceFixer
vf = VoiceFixer()

print('All models successfully downloaded!')
