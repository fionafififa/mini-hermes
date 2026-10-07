import argparse
from urllib.parse import urlsplit

import httpx2

from mini_hermes.config import get_api_key, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    config = load_config("config.anthropic.toml")
    url = urlsplit(config.base_url)
    if url.scheme != "https" or url.hostname != "api.deepseek.com":
        raise ValueError("This diagnostic is only for the official DeepSeek service.")
    api_key = get_api_key(config)

    def show(label, response):
        print(f"\n[{label}]", flush=True)
        print("Status:", response.status_code)
        print("Content-Type:", response.headers.get("content-type"))
        for name in ("request-id", "x-request-id", "x-ds-request-id"):
            if response.headers.get(name):
                print(name + ":", response.headers[name])
        if response.status_code != 200:
            print("Body:", repr(response.text.replace(api_key, "[REDACTED]")[:1500]))

    with httpx2.Client(trust_env=False, timeout=30.0) as client:
        response = client.get(
            "https://api.deepseek.com/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        show("MODELS", response)
        if response.status_code != 200:
            return
        model_ids = [item["id"] for item in response.json()["data"]]
        print("Available models:", model_ids)
        print("Configured model:", config.model)
        if config.model not in model_ids:
            print("MODEL_NOT_LISTED. Check the model ID before further inference calls.")
            return
        if not args.infer:
            print("No generation requested. To compare protocols, rerun with --infer.")
            return

        native_request = {
            "model": config.model,
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
        }
        cases = [
            ("ANTHROPIC_RAW", "https://api.deepseek.com/anthropic/v1/messages",
             {"x-api-key": api_key, "anthropic-version": "2023-06-01"}),
            ("OPENAI_RAW", "https://api.deepseek.com/chat/completions",
             {"Authorization": f"Bearer {api_key}"}),
        ]
        for label, endpoint, headers in cases:
            response = client.post(endpoint, headers=headers, json=native_request)
            show(label, response)


if __name__ == "__main__":
    main()
