"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from homemade import ExampleEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging
from openai import OpenAI
import os


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

# homemade.py (existing imports already at the top)

class LLMEngine(ExampleEngine):
    """A homemade engine that uses an LLM to select moves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = OpenAI(api_key=os.environ["GROQ_API_KEY"],base_url="https://api.groq.com/openai/v1")  # or load from config/env

    def search(self, board, time_limit, ponder, draw_offered, root_moves):
        legal = root_moves if isinstance(root_moves, list) else list(board.legal_moves)
        prompt = f"Position (FEN): {board.fen()}\nChoose the best move in UCI format from: {[m.uci() for m in legal]}"
        response = self.client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[ 
            {"role": "system", "content": "You are a professional chess player. Make the best move"},
            {"role": "user", "content": prompt}
            ],
            temperature=0.2,       # low = more consistent/deterministic moves, less "creative" blundering
            max_tokens=50,         # you just need a short move, no need to allow long rambling
            top_p=0.9,             # optional, works alongside temperature to trim unlikely tokens
            stop=["\n"],           # cuts it off after one line, useful if you're forcing "just the move"
            stream=False,          # set True if you want to stream tokens as they're generated
            seed=42,         )
        move = self._parse_and_validate(response, legal, board)
        return PlayResult(move, None, draw_offered=draw_offered)

    def _parse_and_validate(self, response, legal_moves, board):
        text = response.choices[0].message.content.strip()
        tokens = text.replace(",", " ","|").split()  # split on whitespace, strip stray punctuation
        for move in legal_moves:
            if move.uci() in tokens:
                return move
        logger.info(f"Could not parse LLM move from: {text!r} — playing random move instead")
        return random.choice(legal_moves)