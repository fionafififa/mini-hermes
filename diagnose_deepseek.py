from anthropic import Anthropic, DefaultHttpxClient

from mini_hermes.config import get_api_key, load_config


def main() -> None:
    config = load_config("config.anthropic.toml")
    api_key = get_api_key(config)

    def probe(label: str, trust_env: bool) -> bool:
        print(f"\n[{label}]", flush=True)
        try:
            with Anthropic(
                api_key=api_key,
                base_url=config.base_url,
                timeout=20.0,
                max_retries=0,
                http_client=DefaultHttpxClient(trust_env=trust_env),
            ) as client:
                message = client.messages.create(
                    model=config.model,
                    max_tokens=64,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": "Reply with OK."}],
                    stream=False,
                )
            print("SUCCESS; block types:", [block.type for block in message.content])
            return True
        except Exception as exc:
            print("Error type:", type(exc).__name__)
            response = getattr(exc, "response", None)
            if response is None:
                print("Error:", str(exc).replace(api_key, "[REDACTED]"))
            else:
                print("Status:", response.status_code)
                print("URL:", str(response.request.url).replace(api_key, "[REDACTED]"))
                print("Server:", response.headers.get("server"))
                print("Content-Type:", response.headers.get("content-type"))
                print("Body:", repr(response.text.replace(api_key, "[REDACTED]")[:1500]))
            return False

    if not probe("AUTO_PROXY", True):
        probe("DIRECT", False)


if __name__ == "__main__":
    main()
