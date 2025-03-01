import os
import json
import time
import argparse
import utils
import pipeline
import asyncio
import re

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='', help='The topic of the podcast')
    parser.add_argument('--guest-number', type=int, default=2, help='The number of the guest to be invited to the talk show')
    parser.add_argument('--session-id', type=str, default='', help='Session ID; if empty, a new ID will be allocated')
    return parser.parse_args()
    
async def main():
    args = parse_arguments()

    # Initialize session and API key
    api_key = utils.get_api_key()
    if api_key is None:
        raise ValueError("Please set your OPENAI_KEY in the environment variable.")

    session_id = pipeline.init_session(args.session_id)
    print(f"Session {session_id} is created.")
    
    start_time = time.time()
    await pipeline.full_steps(session_id, args.topic, args.guest_number, api_key)
    end_time = time.time()

    print(f"{session_id} took {end_time - start_time:.2f} seconds to complete.")

if __name__ == "__main__":
    asyncio.run(main())