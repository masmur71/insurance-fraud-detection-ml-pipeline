"""Exercise local MLflow persistence, downloaded artifact identity, and HTTP smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def wait(url, process):
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"Server exited: {process.returncode}")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError(url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5056)
    args = parser.parse_args()
    import mlflow
    from mlflow.tracking import MlflowClient

    store = ROOT / ".mlflow"
    store.mkdir(exist_ok=True)
    uri = f"http://127.0.0.1:{args.port}"
    command = [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1",
               "--port", str(args.port), "--workers", "1", "--backend-store-uri", "sqlite:///.mlflow/tracking.db",
               "--serve-artifacts", "--artifacts-destination", str(store / "artifacts")]
    env = {**os.environ, "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
           "MLFLOW_HTTP_REQUEST_TIMEOUT": "5"}
    with (store / "verification-server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            wait(uri + "/health", server)
            subprocess.run([sys.executable, "scripts/build_model.py", "--output-dir",
                            str(args.output_dir), "--tracking-uri", uri], cwd=ROOT,
                           env=env, check=True)
            client = MlflowClient(tracking_uri=uri)
            experiment = client.get_experiment_by_name("frozen-phase-3-reproduction")
            run = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])[0]
            assert run.info.status == "FINISHED"
            assert len(run.data.params) == 17, run.data.params
            assert len(run.data.metrics) == 11, run.data.metrics
            assert {"source_commit", "working_tree_dirty", "evaluation_scope"} <= run.data.tags.keys()
            assert {item.path for item in client.list_artifacts(run.info.run_id)} == {
                "model.joblib", "build-report.json"}
            with urlopen(uri) as response:
                assert b"html" in response.read().lower()
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"],
                           capture_output=True, check=True)
            server.wait(timeout=20)
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
            wait(uri + "/health", server)
            assert client.get_run(run.info.run_id).data.params == run.data.params
            assert client.get_run(run.info.run_id).data.metrics == run.data.metrics
            target = args.output_dir / "downloaded"
            target.mkdir()
            mlflow.set_tracking_uri(uri)
            for name in ("model.joblib", "build-report.json"):
                downloaded = Path(client.download_artifacts(run.info.run_id, name, str(target)))
                assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == hashlib.sha256(
                    (args.output_dir / name).read_bytes()).hexdigest()
            api_env = {**env, "MODEL_PATH": str((target / "model.joblib").resolve()),
                       "PYTHONPATH": str(ROOT / "src")}
            api = subprocess.Popen([sys.executable, "-m", "uvicorn", "claim_fraud.api:app",
                                    "--host", "127.0.0.1", "--port", "8056"],
                                   cwd=ROOT, env=api_env, stdout=log, stderr=log)
            try:
                wait("http://127.0.0.1:8056/health", api)
                subprocess.run([sys.executable, "scripts/smoke_api.py", "--url",
                                "http://127.0.0.1:8056", "--model", str(target / "model.joblib")],
                               cwd=ROOT, check=True)
            finally:
                api.terminate()
                api.wait(timeout=20)
            evidence = {"run_id": run.info.run_id, "tracking_uri": uri,
                        "artifact_hash_parity": True, "restart_persistence": True,
                        "downloaded_model_http_smoke": True, "ui_http": 200}
            (args.output_dir / "tracking-verification.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(evidence))
        finally:
            if server.poll() is None:
                subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"],
                               capture_output=True, check=True)
                server.wait(timeout=20)


if __name__ == "__main__":
    main()
