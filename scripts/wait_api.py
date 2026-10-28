"""Wait up to 60 seconds for a local container to finish loading its model."""
import argparse
import time
from urllib.error import URLError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urlopen(args.url + "/health", timeout=3) as response:
                if response.status == 200:
                    print("API ready")
                    return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise TimeoutError("API did not become healthy within 60 seconds")


if __name__ == "__main__":
    main()
