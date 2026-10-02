from __future__ import annotations

import argparse
import sys

from step4_ai_agent import AgentConfig, SingleNodeAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-node agent using python-ai-sdk."
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Provider factory name exposed by ai_sdk (e.g. openai, anthropic, google).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id for the selected provider.",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional provider API key. If omitted, provider env var is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()

    agent = SingleNodeAgent(
        AgentConfig(
            provider=args.provider,
            model=args.model,
            system_prompt=args.system,
            api_key=args.api_key,
        )
    )
    response = agent.run(prompt)
    print(response)


if __name__ == "__main__":
    main()
