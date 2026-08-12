"""
Homemade chess engine that uses an LLM (via Groq) to select moves.

The LLM is given the current position and a list of legal moves, and can
call a `check_move_safety` tool to verify whether a candidate move hangs
material before committing to it. This helps catch the "blunders a piece
for no reason" failure mode that plain prompting is prone to.
"""

import json
import logging
import os
import random
from abc import ABC

import chess
from chess.engine import PlayResult, Limit
from openai import OpenAI

from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
# The OpenAI SDK logs every single HTTP handshake step (TLS start, headers
# sent, body sent, etc.) at DEBUG level. When lichess-bot is run with -v,
# that noise buries our own useful debug lines. We turn those three loggers
# down to WARNING so only real problems from them show up, while leaving
# our own `logger` (below) free to print at DEBUG level.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# This is OUR logger for this file. lichess-bot's -v flag controls whether
# logger.debug(...) calls actually get printed; logger.info/warning always
# show. Use logger.debug for verbose "what's happening" traces, and
# logger.warning for things that indicate something went wrong but was
# recovered from (e.g. falling back to a random move).
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BASE CLASS
# ---------------------------------------------------------------------------
# lichess-bot's engine_wrapper.py does `from homemade import ExampleEngine`
# and expects every homemade engine to subclass it. It must be DEFINED here,
# not imported from this same module (that was the original bug — a
# self-import that could never succeed).
class ExampleEngine(MinimalEngine, ABC):
    """Base class for homemade engines."""

    pass


# ---------------------------------------------------------------------------
# MATERIAL VALUES
# ---------------------------------------------------------------------------
# Standard relative piece values, used only by our own safety-check tool
# below (not sent to the LLM). King is intentionally omitted — it's never
# actually captured during legal play, so it has no "value" in this context.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


# ---------------------------------------------------------------------------
# THE SAFETY-CHECK TOOL (the actual logic behind check_move_safety)
# ---------------------------------------------------------------------------
def evaluate_move_safety(board: chess.Board, move: chess.Move) -> dict:
    """Check whether a candidate move hangs material, and what it captures.

    This is a cheap heuristic — comparing how many attackers vs. defenders
    a square has after the move — NOT a full static-exchange evaluation
    (SEE). It won't catch every tactical subtlety (pins, discovered attacks,
    x-ray defenders), but it directly targets the failure mode we saw in
    real games: moving a piece onto a square where it's simply captured for
    free next move (e.g. Bxf7+, Ng5+, Qg4 in the earlier example game).

    Returns a plain dict (not a custom object) because this gets serialized
    to JSON and sent back to the model as a tool result.
    """
    # Work on a copy so we don't mutate the real board the engine is using.
    board_copy = board.copy()

    # If this move is a capture, note the value of what we're taking —
    # useful context for the model even if the move also hangs something.
    captured_value = 0
    if board_copy.is_capture(move):
        captured_piece = board_copy.piece_at(move.to_square)
        if captured_piece is not None:
            captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)

    # Remember whose move this is BEFORE pushing, since board.turn flips
    # after the move is played.
    mover_color = board_copy.turn
    board_copy.push(move)

    # What piece is now sitting on the destination square, and how much is
    # it worth? (After castling/promotion this is still correct — it's
    # whatever ended up on that square.)
    moved_piece = board_copy.piece_at(move.to_square)
    piece_value = PIECE_VALUES.get(moved_piece.piece_type, 0) if moved_piece else 0

    # Who can capture on that square now, and who defends it?
    # attackers = opponent's pieces that could capture there next.
    # defenders = our own pieces that could recapture there.
    attackers = board_copy.attackers(not mover_color, move.to_square)
    defenders = board_copy.attackers(mover_color, move.to_square)

    # Simple rule: if there are strictly more attackers than defenders,
    # treat the piece as hanging. (This under- and over-simplifies real
    # exchange evaluation, but is a fast, good-enough tripwire.)
    hangs = piece_value > 0 and len(attackers) > len(defenders)

    return {
        "move": move.uci(),
        "captures_value": captured_value,
        "hangs_piece": hangs,
        "piece_at_risk_value": piece_value if hangs else 0,
        "gives_check": board_copy.is_check(),
    }


# ---------------------------------------------------------------------------
# TOOL SCHEMA
# ---------------------------------------------------------------------------
# This is the JSON schema Groq's API needs to know the tool exists and how
# to call it. The "description" fields matter — the model decides whether
# and how to call the tool based on this text, so being explicit about WHEN
# to use it (aggressive/capturing moves) steers the model toward using it
# on the moves that actually need checking, rather than every single move.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_move_safety",
            "description": (
                "Check whether one or more candidate legal moves hang material. "
                "You can pass multiple moves at once to compare several candidates "
                "in a single call, which is faster than checking them one by one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more candidate moves in UCI format, e.g. ['e2e4', 'g1f3'].",
                    }
                },
                "required": ["moves"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# THE ENGINE
# ---------------------------------------------------------------------------
class LLMEngine(ExampleEngine):
    """A homemade engine that uses an LLM to select moves, with a move-safety tool."""

    def __init__(self, *args, **kwargs):
        # Pass through to MinimalEngine's __init__ — it sets up things like
        # self.options, self.name, etc. that lichess-bot's wrapper expects.
        super().__init__(*args, **kwargs)

        # Groq exposes an OpenAI-compatible API, so we just point the
        # official openai SDK at Groq's base_url instead of OpenAI's.
        # The API key is read from an environment variable so it never
        # ends up hardcoded or committed to source control.
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

        # --- Tunable knobs, gathered here so they're easy to find/change ---
        self.model = "openai/gpt-oss-120b"
        # "low" was the setting that reliably answered within budget in
        # testing; "medium"/"high" reason more but are prone to running out
        # of max_tokens before producing a final answer (finish_reason=length).
        self.reasoning_effort = "low"
        # Needs headroom for: hidden reasoning tokens + potential tool-call
        # round trips + the final move text.
        self.max_tokens = 800
        # Hard cap on how many times we'll loop asking the model to make
        # tool calls before giving up and falling back to a random move.
        # Prevents an infinite loop if the model keeps calling tools forever.
        self.max_tool_rounds = 4

    def search(
        self,
        board: chess.Board,
        time_limit: Limit,
        ponder: bool,
        draw_offered: bool,
        root_moves: MOVE,
    ) -> PlayResult:
        """Entry point lichess-bot calls every time it's our turn to move."""

        # root_moves is sometimes a pre-filtered list (e.g. from an opening
        # book move restriction) and sometimes not provided at all — in
        # that case, fall back to every legal move in the current position.
        legal = root_moves if isinstance(root_moves, list) else list(board.legal_moves)
        legal_uci = [m.uci() for m in legal]

        # The conversation the model sees. We seed it with a system prompt
        # (behavior instructions) and a user turn (the actual position).
        # This `messages` list will grow as tool calls happen — each tool
        # call and its result gets appended before we ask the model again.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional chess player. You will be given a "
                    "position and a list of legal moves. Before committing to a "
                    "move, use the check_move_safety tool on any move that looks "
                    "aggressive or involves a capture, to make sure it doesn't "
                    "hang material for nothing. Once you have decided, reply with "
                    "ONLY the chosen move in UCI format, nothing else — no "
                    "explanation, no punctuation, just the move."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Position (FEN): {board.fen()}\n"
                    f"Board:\n{board}\n"  # ASCII board — spatial layout can help the model
                    f"Legal moves: {legal_uci}\n"
                    "Pick exactly one move from the Legal moves list above. "
                    "Do not choose any move not in that list."
                ),
            },
        ]

        move = self._run_with_tools(board, legal, legal_uci, messages)

        # If the LLM call failed outright, ran out of tool-call rounds, or
        # gave us something unparseable, we NEVER want to crash the bot or
        # send an illegal move to Lichess — always fall back to something
        # legal so the game keeps going.
        if move is None:
            logger.warning("Falling back to random legal move")
            move = random.choice(legal)

        logger.debug(f"did {move.uci()}")
        return PlayResult(move, None, draw_offered=draw_offered)

    def _run_with_tools(
        self,
        board: chess.Board,
        legal: list,
        legal_uci: list,
        messages: list,
    ):
        """Run the chat/tool-calling loop and return a chess.Move, or None on failure.

        This is the core control loop:
          1. Send the conversation so far to the model.
          2. If the model wants to call a tool, execute it locally and feed
             the result back into the conversation, then loop again (go to 1).
          3. If the model instead returns plain text, try to parse a legal
             move out of it and return that.
          4. Give up after max_tool_rounds iterations, or on any hard error.
        """
        for round_num in range(self.max_tool_rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",  # let the model decide whether to call a tool
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    temperature=0.2,  # low = more consistent, less "creative" move choice
                    top_p=0.9,
                    seed=42,  # deterministic sampling where the backend supports it
                    stream=False,  # we want the full response at once, not token-by-token
                )
            except Exception as e:
                # Covers network errors, malformed-request errors, etc.
                # We don't want a transient API hiccup to crash the whole bot process.
                logger.warning(f"LLM request failed: {e}")
                return None

            choice = response.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason
            logger.debug(f"round {round_num}: finish_reason={finish_reason}")

            # tool_calls is populated when the model decided to call one or
            # more tools instead of (or before) answering directly.
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
                # Record the assistant's tool-call turn in the conversation
                # history (required by the API — the model needs to see its
                # own prior tool-call message before it can see the results).
                messages.append(msg)

                # A single model turn can request multiple tool calls at
                # once (e.g. checking two different candidate moves) — we
                # must answer every one of them before continuing.
                for call in tool_calls:
                    result = self._dispatch_tool(board, call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,  # links result back to the specific call
                            "content": json.dumps(result),
                        }
                    )
                # Loop back to the top: ask the model again now that it has
                # the tool results in context. It might call more tools, or
                # give its final answer this time.
                continue

            # No tool call this round — the model should be giving its
            # final move as plain text.
            text = (msg.content or "").strip()

            if not text:
                # Distinguish "ran out of tokens mid-thought" from other
                # empty-response causes, since they need different fixes
                # (raise max_tokens vs. investigate something else).
                if finish_reason == "length":
                    logger.warning(
                        "Empty LLM response (finish_reason=length) — "
                        "ran out of tokens before answering"
                    )
                else:
                    logger.warning(f"Empty LLM response (finish_reason={finish_reason})")
                return None

            move = self._parse_move(text, legal, legal_uci)
            if move is not None:
                return move

            # The model answered with something, but we couldn't find a
            # legal move inside it (e.g. it hallucinated a move that isn't
            # in the position, or wrote a full sentence instead of a move).
            logger.info(f"Could not parse LLM move from: {text!r}")
            return None

        # We used up all our tool-call rounds without getting a final
        # answer — treat this the same as any other failure.
        logger.warning(f"Exceeded {self.max_tool_rounds} tool-calling rounds without an answer")
        return None

    def _dispatch_tool(self, board: chess.Board, call) -> dict:
        try:
            args = json.loads(call.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            return {"error": "could not parse tool arguments"}

        if call.function.name == "check_move_safety":
            moves_uci = args.get("moves", [])
            results = []
            for move_uci in moves_uci:
                try:
                    move = chess.Move.from_uci(move_uci)
                except ValueError:
                    results.append({"move": move_uci, "error": "invalid UCI format"})
                    continue
                if move not in board.legal_moves:
                    results.append({"move": move_uci, "error": "not a legal move here"})
                    continue
                results.append(evaluate_move_safety(board, move))
            logger.debug(f"tool check_move_safety({moves_uci}) -> {results}")
            return {"results": results}

        return {"error": f"unknown tool '{call.function.name}'"}

    @staticmethod
    def _parse_move(text: str, legal: list, legal_uci: list):
        """Try to find a legal move inside the model's free-text answer.

        Even with strong prompting, models sometimes wrap the move in extra
        punctuation or a short phrase ("Move: e2e4", "e2e4.", etc). Rather
        than requiring an exact match, we split on common separators and
        check every resulting token against the legal-move list — this is
        more forgiving without being so loose that it accepts nonsense.
        """
        tokens = text.replace(",", " ").replace("|", " ").replace(".", " ").split()

        for token in tokens:
            cleaned = token.strip().lower()
            if cleaned in legal_uci:
                # Return the actual chess.Move object (not just the string)
                # since that's what PlayResult / lichess-bot expects.
                return legal[legal_uci.index(cleaned)]

        return None