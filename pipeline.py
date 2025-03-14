import datetime
import os
from string import Template
from openai import OpenAI, AsyncOpenAI
import re
import glob
import pickle
import time
import json5
from retrying import retry
from code_generator import collect_and_check_audio_data
from utils import check_conversation_script, check_json_script, check_map
import random
import string
import asyncio
import tiktoken

import utils
import voice_presets
from code_generator import AudioCodeGenerator

# Enable this for debugging
USE_OPENAI_CACHE = os.environ.get('USE_OPENAI_CACHE', False)
openai_cache = []
if USE_OPENAI_CACHE:
    os.makedirs('cache', exist_ok=True)
    for cache_file in glob.glob('cache/*.pkl'):
        with open(cache_file, 'rb') as file:
            openai_cache.append(pickle.load(file))

def chat_with_gpt(prompt, api_key, sys_info="You are a helpful assistant."):
    if USE_OPENAI_CACHE:
        filtered_object = list(filter(lambda x: x['prompt'] == (sys_info + prompt), openai_cache))
        if len(filtered_object) > 0:
            response = filtered_object[0]['response']
            return response
    
    client = OpenAI(api_key=api_key, timeout=240.0, max_retries=0)
    chat = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": sys_info
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    if USE_OPENAI_CACHE:
        cache_obj = {
            'prompt': sys_info + prompt,
            'response': chat.choices[0].message.content
        }
        cache_file = f'cache/{time.time()}.pkl'
        with open(cache_file, 'wb') as _openai_cache:
            pickle.dump(cache_obj, _openai_cache)
            openai_cache.append(cache_obj)

    return chat.choices[0].message.content

async def achat_with_gpt(prompt, api_key, sys_info="You are a helpful assistant."):
    if USE_OPENAI_CACHE:
        filtered_object = list(filter(lambda x: x['prompt'] == (sys_info + prompt), openai_cache))
        if len(filtered_object) > 0:
            response = filtered_object[0]['response']
            return response

    client = AsyncOpenAI(api_key=api_key)
    chat = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": sys_info
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    if USE_OPENAI_CACHE:
        cache_obj = {
            'prompt': sys_info + prompt,
            'response': chat.choices[0].message.content
        }
        with open(f'cache/{time.time()}.pkl', 'wb') as _openai_cache:
            pickle.dump(cache_obj, _openai_cache)
            openai_cache.append(cache_obj)

    return chat.choices[0].message.content

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    token_count = len(tokens)
    
    return token_count

def get_file_content(filename):
    with open(filename, 'r') as file:
        return file.read().strip()


def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def extract_substring_with_quotes(input_string, quotes="'''"):
    pattern = f"{quotes}(.*?){quotes}"
    matches = re.findall(pattern, input_string, re.DOTALL)
    for i in range(len(matches)):
        if matches[i][:4] == 'json':
            matches[i] = matches[i][4:]
    
    if len(matches) == 1:
        return matches[0]
    else:
        return matches


def try_extract_content_from_quotes(content):
    if "'''" in content:
        return extract_substring_with_quotes(content)
    elif "```" in content:
        return extract_substring_with_quotes(content, quotes="```")
    else:
        return content

def maybe_get_content_from_file(content_or_filename):
    if os.path.exists(content_or_filename):
        with open(content_or_filename, 'r') as file:
            return file.read().strip()
    return content_or_filename

def detect_language(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    english_pattern = re.compile(r'[a-zA-Z]')

    chinese_count = len(chinese_pattern.findall(text))
    english_count = len(english_pattern.findall(text))

    if chinese_count > english_count:
        return "zh"
    else:
        return ""

def init_session(session_id=''):
    def uid8():
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    if session_id == '':
        session_id = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{uid8()}'

    # create the paths
    os.makedirs(utils.get_session_voice_preset_path(session_id), exist_ok=True)
    os.makedirs(utils.get_session_audio_path(session_id), exist_ok=True)
    print(f'New session created, session_id={session_id}')
    return session_id

@retry(stop_max_attempt_number=3)
def input_text_to_json_script_with_retry(complete_prompt_path, conversation_script, api_key):
    print("    trying ...")
    sys_info = "You are a helpful audio script writer."
    complete_prompt = get_file_content(complete_prompt_path)
    json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt, api_key, sys_info))

    try:
        json_data = json5.loads(json_response)
        json_data = check_json_script(json_data, conversation_script)
        collect_and_check_audio_data(json_data)
    except Exception as err:
        print(f'JSON ERROR: {err}')
        retry_complete_prompt = f'{complete_prompt}\n```\n{json_response}```\nThe script above has format error: {err}. Return the fixed script.\n\nScript:\n'
        write_to_file(complete_prompt_path, retry_complete_prompt)
        raise err

    return json_data

@retry(stop_max_attempt_number=3)
def format_script_with_retry(response, requirement, api_key):
    print("    trying ...")
    try:
        complete_prompt = f'```\n{response}```\nThe script above has format error, here is the format requirement: {requirement}. Please return the fixed script.\n\nScript:\n'
        json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt, api_key))
        json_data = json5.loads(json_response)
    except Exception as err:
        print(f'JSON ERROR: {err}')
        response = json_response
        raise err

    return json_data

def generate_guest_info(lang, guest_number, topic, output_path, api_key):
    print("【Step1】Generating guest info ...")
    
    if lang == 'zh':
        complete_prompt_path = f'prompts/Step1_organizer_zh.prompt'
    else:
        complete_prompt_path = f'prompts/Step1_organizer.prompt'
    complete_prompt = get_file_content(complete_prompt_path)
    complete_prompt = complete_prompt.replace('${guest_number_to_be_replaced}', str(guest_number))
    complete_prompt = complete_prompt.replace('${topic_to_be_replaced}', topic)

    json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt, api_key))
    try:
        response_json = json5.loads(json_response)
    except Exception as err:
        requirement = f'The script should in json format but with this error: {err}'
        json_response = format_script_with_retry(json_response, requirement, api_key)
        response_json = json5.loads(json_response)

    guest_info = {key: value for key, value in response_json.items() if key.startswith('Guest_')}
    outline_info = response_json.get('Interview_Outline', [])

    save_path = f'{output_path}/Step1_output.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json5.dump(response_json, f, ensure_ascii=False, indent=4)

    return guest_info, outline_info

async def collect_guest_response(lang, guest_number, topic, guest_info, outline_info, output_path, api_key):
    print("【Step2】Collecting guest response ...")

    if lang == 'zh':
        sys_prompt_path = f'prompts/Step2_guest_sys_zh.prompt'
        usr_prompt_path = f'prompts/Step2_guest_usr_zh.prompt'
    else:
        sys_prompt_path = f'prompts/Step2_guest_sys.prompt'
        usr_prompt_path = f'prompts/Step2_guest_usr.prompt'
    
    sys_info = get_file_content(sys_prompt_path)
    complete_prompt = get_file_content(usr_prompt_path)
    
    max_word_number = 1000
    complete_prompt = complete_prompt.replace('${topic_to_be_replaced}', topic)
    complete_prompt = complete_prompt.replace('${max_word_number}', str(max_word_number))
    complete_prompt = complete_prompt.replace('${outline_to_be_replaced}', json5.dumps(outline_info, ensure_ascii=False))
    
    tasks = []
    for i in range(1, guest_number + 1):
        sys_prompt = sys_info.replace('${guest_info_to_be_replaced}', json5.dumps(guest_info[f'Guest_{i}'], ensure_ascii=False))
        tasks.append(achat_with_gpt(complete_prompt, api_key, sys_prompt))
    collections = await asyncio.gather(*tasks)

    collected_data = []
    for i, response in enumerate(collections):
        collected_data.append(f"\'\'\'\n{guest_info[f'Guest_{i + 1}']}\n{try_extract_content_from_quotes(response)}\n\'\'\'")
    collected_data_str = '\n'.join(collected_data)

    save_path = output_path / 'Step2_output.json'
    write_to_file(save_path, collected_data_str)
    
    return collected_data_str

def generate_conversation_script(lang, guest_number, collected_data_str, topic, output_path, api_key):
    print("【Step3】Generating conversation script ...")
    if lang == 'zh':
        complete_prompt_path = f'prompts/Step3_organizer_zh.prompt'
    else:
        complete_prompt_path = f'prompts/Step3_organizer.prompt'

    complete_prompt = get_file_content(complete_prompt_path)
    
    complete_prompt = complete_prompt.replace('${guest_number_to_be_replaced}', str(guest_number))
    complete_prompt = complete_prompt.replace('${topic_to_be_replaced}', topic)
    complete_prompt = complete_prompt.replace('${collected_data_to_be_replaced}', collected_data_str)

    json_response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt, api_key))
    try:
        json_data = json5.loads(json_response)
        check_conversation_script(json_data)
    except Exception as err:
        requirement = 'The script should in json format as following: [{"speaker":..., "speaking_content":..., "speaking_style":...}, {...},...]'
        json_data = format_script_with_retry(json_response, requirement, api_key)

    generated_conv_script_filename = output_path / 'conversation_script.json'
    write_to_file(generated_conv_script_filename, '\n'.join([json5.dumps(item, ensure_ascii=False) for item in json_data]))

    return json_data


# Step 1: conversation script to audio script
@retry(stop_max_attempt_number=3)
def conversation_to_audio_script(conversation_script, output_path, api_key):
    print("【Step4】Generating audio script ...")
    prompt_template = f'prompts/Step4_organizer.prompt'
    conv_to_audio_script_prompt = get_file_content(prompt_template)
    prompt = f'{conv_to_audio_script_prompt}\n\n[Conversation script]: \n\'\'\'\n{conversation_script}\n\'\'\'\n[Audio Script]:\n'

    last_response_file = output_path / 'last_conv_text_to_audio_script.response'
    if os.path.exists(last_response_file):
        last_response = get_file_content(last_response_file)
        prompt = f'{prompt}\n{last_response}'

    print("    trying ...")
    sys_info = "You are a helpful audio script writer."
    json_response = try_extract_content_from_quotes(chat_with_gpt(prompt, api_key, sys_info))

    try:
        json_data = json5.loads(json_response)
        json_data = check_json_script(json_data, conversation_script)
        collect_and_check_audio_data(json_data)
    except Exception as err:
        print(f'    ERROR: {err}')
        retry_complete_prompt = f'```\n{json_response}```\nThe audio script above has format error: {err}. Return the fixed script.\n\nScript:\n'
        write_to_file(last_response_file, retry_complete_prompt)
        raise err

    generated_audio_script_filename = output_path / 'audio_script.json'
    write_to_file(generated_audio_script_filename, '\n'.join([json5.dumps(item, ensure_ascii=False) for item in json_data]))
    return json_data


def role_to_voice_map(lang, voices, output_path, api_key):
    print(f'【Step5】Parsing character voice with LLM...')
    if lang == 'zh':
        prompt = get_file_content('prompts/bginfo_to_voice_map_zh.prompt')
    else:
        prompt = get_file_content('prompts/bginfo_to_voice_map.prompt')

    bg_info_file = get_file_content(f'{output_path}/Step1_output.json')
    bg_info = json5.loads(bg_info_file)

    prompt = prompt.replace('${info_to_be_replaced}', json5.dumps(bg_info, ensure_ascii=False))
    
    presets = '\n'.join(f"{preset['id']}: {preset['desc']}" for preset in voices.values())
    prompt = prompt.replace('${voice_and_desc}', json5.dumps(presets, ensure_ascii=False))

    write_to_file(output_path / 'complete_info_to_char_voice_map.prompt', prompt)
    char_voice_map_response = try_extract_content_from_quotes(chat_with_gpt(prompt, api_key)) 

    try:
        char_voice_map = json5.loads(char_voice_map_response)
        char_voice_map = check_map(char_voice_map, bg_info)
    except Exception as err:
        requirement = 'The script should in json format as {"":"", "":"", ...}.'
        char_voice_map_response = format_script_with_retry(char_voice_map_response, requirement, api_key)
        char_voice_map = json5.loads(char_voice_map_response)
    
    # enrich char_voice_map with voice preset metadata
    complete_char_voice_map = {c: voices[char_voice_map[c]] for c in char_voice_map}
    char_voice_map_filename = output_path / 'character_voice_map.json'
    write_to_file(char_voice_map_filename, json5.dumps(complete_char_voice_map, ensure_ascii=False))
    return complete_char_voice_map


def json_script_and_char_voice_map_to_audio_gen_code(json_script_filename, char_voice_map_filename, output_path, result_filename):
    print(f'【Step6】Compiling audio script to Python program ...')
    audio_code_generator = AudioCodeGenerator()
    code = audio_code_generator.parse_and_generate(
        json_script_filename,
        char_voice_map_filename,
        output_path,
        result_filename
    )
    write_to_file(output_path / 'audio_generation.py', code)

# Step 4: py code to final wav
def audio_code_gen_to_result(audio_gen_code_path):
    print(f'【Step7】Start running Python program ...')
    audio_gen_code_filename = audio_gen_code_path / 'audio_generation.py'
    os.system(f'PYTHONPATH=. python {audio_gen_code_filename}')


async def generate_json_file(session_id, lang, topic, guest_number, api_key):
    output_path = utils.get_session_path(session_id)
    
    # Get response from the host agent
    guest_info, outline_info = generate_guest_info(lang, guest_number, topic, output_path, api_key)
    # Get responses from guest agents
    guest_responses = await collect_guest_response(lang, guest_number, topic, guest_info, outline_info, output_path, api_key)
    # Get conversation script from the writer agent
    conversation_script = generate_conversation_script(lang, guest_number, guest_responses, topic, output_path, api_key)
    # Get audio script from the writer agent
    audio_script = conversation_to_audio_script(conversation_script, output_path, api_key)

    return audio_script

def generate_audio(session_id, lang, json_script, api_key):

    max_lines = utils.get_max_script_lines()
    if len(json_script) > max_lines:
        raise ValueError(f'The number of lines of the JSON script has exceeded {max_lines}!')

    output_path = utils.get_session_path(session_id)
    output_audio_path = utils.get_session_audio_path(session_id)
    
    # role-voice matching
    voices = voice_presets.get_merged_voice_presets(session_id, lang)
    char_voice_map = role_to_voice_map(lang, voices, output_path, api_key)
    
    # code generation
    json_script_filename = output_path / 'audio_script.json'
    char_voice_map_filename = output_path / 'character_voice_map.json'
    result_wav_basename = f'res_{"_".join(session_id.split("/"))}'
    json_script_and_char_voice_map_to_audio_gen_code(json_script_filename, char_voice_map_filename, output_path, result_wav_basename)
    
    # code running
    audio_code_gen_to_result(output_path)

    result_wav_filename = output_audio_path / f'{result_wav_basename}.wav'
    print(f'Done all processes, session_id={session_id}, result: {result_wav_filename}')
    return result_wav_filename, char_voice_map

async def full_steps(session_id, topic, guest_number, api_key):
    lang = detect_language(topic)
    if lang:
        print('Language: Chinese.')
    else:
        print('Language: English.')
    
    #Generate Scripts
    json_script = await generate_json_file(session_id, lang, topic, guest_number, api_key)

    #Generate Audios
    _, _ = generate_audio(session_id, lang, json_script, api_key)
