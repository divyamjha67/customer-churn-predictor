"""Download the upstream, commit-pinned IBM benchmark with provenance."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
URL = f"https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/{COMMIT}/data/Telco-Customer-Churn.csv"

def main():
    destination = ROOT / "data/raw/telco.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    manifest_path = ROOT / "data/source.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())["sha256"]
        if digest != expected:
            raise RuntimeError("Dataset checksum changed; investigate upstream before training")
    destination.write_bytes(content)
    manifest_path.write_text(json.dumps({
        "source": URL, "commit": COMMIT, "sha256": digest,
        "upstream_repository_license": "Apache-2.0",
        "description": "IBM fictional telecom sample; not production customer observations"
    }, indent=2) + "\n")
    print(f"Downloaded {len(content):,} bytes. SHA256 {digest}")

if __name__ == "__main__":
    main()
