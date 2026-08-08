# LLM Chess Bot on Lichess — Setup Checklist

Using `lichess-bot` + an LLM (OpenRouter or Grok/xAI) as the "engine."

---

## 1. Lichess Account & Token

- [ ] Create a **brand new** Lichess account dedicated to the bot (never use your main/personal account — bot conversion is irreversible and changes what the account can do)
- [ ] Log into that account
- [ ] Go to **Account Settings → API access tokens** (or `lichess.org/account/oauth/token`)
- [ ] Create a new **personal access token**
- [ ] Select scope: `bot:play` (this is the only scope lichess-bot strictly needs)
- [ ] Copy the token somewhere safe — Lichess only shows it once
- [ ] Do **not** commit this token to git or share it publicly

---

## 2. Local Environment Setup

- [ ] Install **Python 3.10+** (check with `python3 --version`)
- [ ] Install **git**
- [ ] Clone the repo: `lichess-bot-devs/lichess-bot`
- [ ] Create a virtual environment (`venv` or `conda`) so dependencies don't clash with other projects
- [ ] Activate the virtual environment
- [ ] Install dependencies from `requirements.txt`
- [ ] Copy `config.yml.default` → `config.yml` (this is the file you'll edit, never the `.default` one)

---

## 3. Configure the Token

- [ ] Open `config.yml`
- [ ] Set the `token` field to your Lichess OAuth token
  - Hint: better practice is to store it in an environment variable and reference it, rather than pasting it in plain text, in case you ever share the config file
- [ ] Leave the rest of `config.yml` at defaults for now — you'll edit the `engine` section later

---

## 4. Upgrade Account to BOT Status

- [ ] Follow the wiki page **"Upgrade to a BOT account"**
- [ ] Run the provided upgrade command/script using your token
- [ ] Confirm on lichess.org that the account now shows a **BOT** tag next to its username
- [ ] Note: after this, the account can no longer play normal rated games against humans outside the Bot API — this is permanent

---

## 5. Get API Access to Your LLM

- [ ] **If using OpenRouter:**
  - [ ] Create an account at openrouter.ai
  - [ ] Generate an API key
  - [ ] Pick a model string (e.g. a specific model slug) — note OpenRouter lets you swap models later just by changing this string
  - [ ] Check pricing/rate limits for your chosen model — chess bots can burn through a lot of calls fast
- [ ] **If using Grok (xAI):**
  - [ ] Create an xAI developer account
  - [ ] Generate an API key
  - [ ] Note the model name/version you'll target
- [ ] Store the API key as an **environment variable** (e.g. `OPENROUTER_API_KEY` or `XAI_API_KEY`) — never hardcode it
- [ ] Test the key works with a simple standalone API call (outside lichess-bot) before wiring it in

---

## 6. Build the "Homemade" Engine Wrapper

- [ ] Open `homemade.py` in the repo
- [ ] Create a new class that subclasses lichess-bot's homemade engine base
- [ ] Inside it, on each move request:
  - [ ] Get the current board state (FEN) from the game object
  - [ ] Use `python-chess` (already a dependency) to compute the **list of legal moves** in that position
  - [ ] Build a prompt containing: the FEN, a human-readable move list (SAN), and clear instructions to output **exactly one move from the list**
  - [ ] Call your LLM API (OpenRouter/Grok) with that prompt
  - [ ] Parse the model's text response to extract the chosen move
- [ ] Keep the prompt short and structured — long chain-of-thought responses slow you down and are harder to parse reliably

---

## 7. Add a Legality Safety Net

- [ ] After parsing the LLM's move, **validate it against the actual legal move list** using `python-chess`
- [ ] If invalid or unparseable:
  - [ ] Retry once with a stricter/simplified prompt (e.g. "Reply with ONLY the move, nothing else")
  - [ ] If it fails again, fall back to a default (random legal move, or a simple heuristic like "prefer captures") so the bot never times out or forfeits
- [ ] Log every fallback event — frequent fallbacks mean your prompt needs work

---

## 8. Handle Latency & Time Controls

- [ ] Set a **timeout** on your LLM API call (e.g. 10–15 sec) with a fallback move if it's exceeded
- [ ] In `config.yml`, restrict matchmaking to slower time controls first (rapid/classical/correspondence, not bullet/blitz)
- [ ] Consider adding a small delay/queue if you expect to hit API rate limits during a game

---

## 9. Point config.yml at Your Engine

- [ ] In `config.yml`, under the `engine` section, set it to use your new Homemade class instead of an external UCI engine (e.g. Stockfish)
- [ ] Double-check `protocol`/`engine_type` settings match what lichess-bot expects for homemade engines (check the wiki's "Create a homemade engine" page)
- [ ] Set `matchmaking` settings conservatively at first (off, or narrow rating range) so you're not immediately flooded with games

---

## 10. Logging for Future Improvement

- [ ] Decide on a log format now (JSON lines is easiest): one entry per move with
  - [ ] FEN before move
  - [ ] legal move list given to the model
  - [ ] raw LLM response
  - [ ] parsed/validated move
  - [ ] whether a fallback was triggered
  - [ ] game ID, timestamp
- [ ] After each game ends, also log the **final result** (win/loss/draw) tied to that game ID
- [ ] Write logs to a local file or lightweight database (SQLite is fine to start)
- [ ] This log becomes your dataset for prompt iteration, fine-tuning, or building an eval set later

---

## 11. Local Testing Before Going Live

- [ ] Run the bot locally: `python lichess-bot.py`
- [ ] From a second throwaway Lichess account, send your bot a challenge
- [ ] Watch console logs during the game for:
  - [ ] API errors/timeouts
  - [ ] Illegal move fallbacks
  - [ ] Crashes on edge cases: checkmate, stalemate, promotion, draw offers, takeback requests
- [ ] Play through a few full games manually before enabling matchmaking or tournaments

---

## 12. Go Live Carefully

- [ ] Enable casual game acceptance first (not rated, not tournaments)
- [ ] Monitor rating, error logs, and API spend for the first several games
- [ ] Gradually enable rated play / matchmaking / tournaments once stable
- [ ] Expect a low rating initially — LLMs are typically weak at tactics without a search layer on top; this is normal, not a bug

---

## Optional Next Steps (Self-Improvement Loop)

- [ ] Periodically review logged games for blunders or fallback spikes
- [ ] Iterate on the prompt (e.g. adding simple heuristics or evaluation hints)
- [ ] If fine-tuning is in scope: curate a dataset from your logs (position → good move) and fine-tune a smaller/cheaper model
- [ ] Consider A/B testing model versions by swapping the OpenRouter model string and comparing win rate over N games
