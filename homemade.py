"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import logging


# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

# homemade.py (existing imports already at the top)

class LLMEngine(ExampleEngine):
    """A homemade engine that uses an LLM to select moves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = SomeLLMClient(api_key="...")  # or load from config/env

    def search(self, board, time_limit, ponder, draw_offered, root_moves):
        legal = root_moves if isinstance(root_moves, list) else list(board.legal_moves)
        prompt = f"Position (FEN): {board.fen()}\nChoose the best move in UCI format from: {[m.uci() for m in legal]}"
        response = self.client.query(prompt)
        move = self._parse_and_validate(response, legal, board)
        return PlayResult(move, None, draw_offered=draw_offered)

    def _parse_and_validate(self, response, legal_moves, board):
        suggested_moves = [for move in board.legal_moves if move in response return move]
        print(suggested_moves)