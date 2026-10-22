# Contributing

Thank you for helping. Bug reports, questions and small pull requests are
all welcome.

## Set up

```bash
git clone https://github.com/prashant7799/barrierscan
cd barrierscan
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Before you open a pull request

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format .
```

## Rules

- Any change to the labelling rules must change both the kernel
  (`src/barrierscan/_kernel.py`) and the oracle (`src/barrierscan/oracle.py`),
  and the oracle tests must still pass on every row.
- Keep the oracle slow and obvious. It is the reference, not a second fast
  path.
- Add a small hand case for every new rule.
- Use synthetic data only in tests and examples. Never commit real data,
  keys or credentials.
- If a README output changes, update the README. A test checks it.
- Write plain code with clear names. A plain loop is better than a clever
  one-liner.

## Reporting a bug

Please include a small synthetic input that shows the problem, the call you
made, what you expected, and what you got. If you can, include the output
of `brute_force_one` for the row in question.
