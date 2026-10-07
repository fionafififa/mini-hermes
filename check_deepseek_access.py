"""Compare configured credentials and HTTP clients using GET /models only."""

import os
from urllib.parse import urlsplit

import httpx
import httpx2

from mini_hermes.config import load_config


def main() -> None:
    credentials = []
    for path in ("config.toml", "config.anthropic.toml"):
        config = load_config(path)
        url = urlsplit(config.base_url)
        if url.scheme != "https" or url.hostname != "api.deepseek.com":
            raise ValueError("Only official DeepSeek configurations are allowed.")
        key = os.environ.get(config.api_key_env, "").strip()
        print(f"{path}: env={config.api_key_env}, key_present={bool(key)}")
        credentials.append((path, key))

    first, second = [key for _, key in credentials]
    print("KEYS_MATCH:", first == second if first and second else "NOT_COMPARABLE")

    unique_credentials = []
    for label, key in credentials:
        if key and all(key != previous for _, previous in unique_credentials):
            unique_credentials.append((label, key))
    if not unique_credentials:
        print("No API key is set in this terminal. No requests sent.")
        return

    for label, key in unique_credentials:
        for name, module in (("httpx", httpx), ("httpx2", httpx2)):
            print(f"\n[{label} / {name} / DIRECT]", flush=True)
            try:
                with module.Client(trust_env=False, timeout=20.0) as client:
                    response = client.get(
                        "https://api.deepseek.com/models",
                        headers={"Authorization": f"Bearer {key}"},
                    )
            except (httpx.HTTPError, httpx2.HTTPError) as exc:
                print("Network error type:", type(exc).__name__)
                continue

            print("Status:", response.status_code)
            print("Content-Type:", response.headers.get("content-type"))
            body = response.text
            for _, configured_key in unique_credentials:
                body = body.replace(configured_key, "[REDACTED]")
            print("Body:", repr(body[:1500]))


if __name__ == "__main__":
    main()
