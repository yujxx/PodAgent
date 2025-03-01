conda create -n podcast -y python=3.10 && \
conda run --live-stream -n podcast conda install -y -c conda-forge pynini==2.1.5 && \
conda run --live-stream -n podcast pip install -r TTS/CosyVoice/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com && \
conda run --live-stream -n podcast pip install -U git+https://git@github.com/facebookresearch/audiocraft@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft && \
conda run --live-stream -n podcast pip install pip==23.2.1 && \
conda run --live-stream -n podcast pip install -r requirements.txt
