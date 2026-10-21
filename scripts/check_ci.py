"""Read GitHub Actions for the current commit using existing Git authentication.

Credentials stay in process memory and are never printed, persisted, or sent outside
api.github.com. This command only reads repository workflow/job metadata.
"""

import argparse
import json
import os
import subprocess
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPOSITORY = "Paarthexe/pico"


def credentials() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        timeout=15,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = values.get("password")
    if not token:
        raise RuntimeError("GitHub authentication is unavailable; authenticate Git or set GH_TOKEN.")
    return token


def request(path: str, token: str) -> dict:
    req = Request(
        f"https://api.github.com/repos/{REPOSITORY}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Pico-validation",
        },
    )
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read(5 * 1024 * 1024))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", default="HEAD")
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", args.sha], text=True).strip()
    token = credentials()
    try:
        runs = request(f"actions/runs?head_sha={sha}&per_page=10", token)["workflow_runs"]
        output = []
        for run in runs:
            jobs = request(f"actions/runs/{run['id']}/jobs?per_page=100", token)["jobs"]
            output.append(
                dict(
                    id=run["id"],
                    name=run["name"],
                    sha=run["head_sha"],
                    status=run["status"],
                    conclusion=run["conclusion"],
                    url=run["html_url"],
                    jobs=[
                        dict(
                            id=j["id"],
                            name=j["name"],
                            status=j["status"],
                            conclusion=j["conclusion"],
                            url=j["html_url"],
                            failed_steps=[s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"],
                        )
                        for j in jobs
                    ],
                )
            )
        print(json.dumps(dict(sha=sha, runs=output), indent=2))
    except HTTPError as exc:
        raise SystemExit(f"GitHub API returned HTTP {exc.code}; no credentials were printed.") from None


if __name__ == "__main__":
    main()
