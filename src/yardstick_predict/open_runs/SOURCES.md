# Where Open Runs comes from

`runs.jsonl.xz` holds public coding-agent runs in yardstick-predict's runs format. They come from these collections. Each keeps its own license.

1. **Terminal-Bench 2.0 trajectories**, gathered by Yoonho Lee. https://huggingface.co/datasets/yoonholee/terminalbench-trajectories. Apache License 2.0. Runs with ids that start with `tb-`.
2. **Terminal-Bench 2.0 leaderboard**, by the Harbor Framework and the agents that submitted to it. https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard. Apache License 2.0. Runs with ids that start with `tbl-`.
3. **SWE-bench Lite agent traces**, gathered by Melissa Pan. https://huggingface.co/datasets/melissapan/swe-bench-lite-agent-traces-v14. Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/. Runs with ids that start with `swe-`.

The Terminal-Bench prompts are the tasks' instructions from https://huggingface.co/datasets/harborframework/terminal-bench-2.0 and https://huggingface.co/datasets/harborframework/terminal-bench-2.1, under the Apache License 2.0. The SWE-bench Lite prompts are the first message each run's agent got: the text of a public GitHub issue, from SWE-bench Lite.

The Apache License 2.0 is in the `LICENSE` file that comes with yardstick-predict.

## What was changed

- Each run keeps only its prompt, AI model, agent, start time, seconds, tokens and cost. Everything else is left out, such as each step the agent took.
- A Terminal-Bench run's seconds are its agent's own working time. A SWE-bench Lite run's seconds come from its record. For Claude Code, they're Claude Code's own count. For Pi, they're the time from the first event to the last.
- Runs whose time or tokens weren't measured are left out. So are the SWE-bench Lite runs by Codex, whose records give no times.
- AI model names keep only the AI model's own name. `anthropic/claude-opus-4-7` becomes `claude-opus-4-7`. Claude models are written with dashes, as Claude Code writes them.
- In the prompts, email addresses become `[email]`. A home folder's user name becomes `me`. Anything shaped like a key becomes `[hidden]`.
- At most two runs of one prompt by one AI model and agent are kept.

`vectors.sqlite` holds Voyage embeddings of the prompts and of the descriptions in `answers.json`. `answers.json` holds what LLM analysis said about each prompt. Both were made by yardstick-predict's own scripts.
