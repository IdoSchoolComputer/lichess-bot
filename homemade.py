"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from abc import ABC


class ExampleEngine(MinimalEngine, ABC):
    """Base class for homemade engines."""

    pass
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
        prompt = (
       f"Position (FEN): {board.fen()}\n"
       f"Legal moves: {[m.uci() for m in legal]}\n"
       f"Pick exactly one move from the Legal moves list above. "
       f"Do not choose any move not in that list."
       )
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a professional chess player. Reply with ONLY the move in UCI format, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            top_p=0.9,
            stream=False,
            seed=42,
            reasoning_effort="medium"
        )
        
        move = self._parse_and_validate(response, legal, board)
        logger.debug(f"did {move}")
        return PlayResult(move, None, draw_offered=draw_offered)

    def _parse_and_validate(self, response, legal_moves, board):
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        finish_reason = choice.finish_reason

        if not text:
            logger.warning(f"Empty LLM response (finish_reason={finish_reason}) — playing random move instead")
            return random.choice(legal_moves)

        tokens = text.replace(",", " ").replace("|", " ").split()
        for move in legal_moves:
            if move.uci() in tokens:
                return move

        logger.info(f"Could not parse LLM move from: {text!r} — playing random move instead")
        return random.choice(legal_moves)