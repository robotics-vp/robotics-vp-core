#!/usr/bin/env python3
"""Create or update the single nightly audit status issue."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


MARKER_PREFIX = "<!-- economic-world-model-audit:digest="


class GitHubIssueClient:
    def __init__(self, token: str, repository: str) -> None:
        self.token = token
        self.repository = repository
        self.base_url = f"https://api.github.com/repos/{repository}"

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                content = response.read().decode("utf-8")
                return json.loads(content) if content else None
        except urllib.error.HTTPError as exc:  # pragma: no cover - network-only path
            raise RuntimeError(f"GitHub API error {exc.code}: {exc.read().decode('utf-8')}") from exc

    def list_open_issues(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "issues?state=open&per_page=100")
        return [row for row in data if "pull_request" not in row]

    def create_issue(self, title: str, body: str) -> Dict[str, Any]:
        return self._request("POST", "issues", {"title": title, "body": body})

    def update_issue(self, issue_number: int, body: str) -> Dict[str, Any]:
        return self._request("PATCH", f"issues/{issue_number}", {"body": body})

    def add_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        return self._request("POST", f"issues/{issue_number}/comments", {"body": body})


def _extract_digest(body: str) -> Optional[str]:
    match = re.search(r"<!-- economic-world-model-audit:digest=([a-f0-9]+) -->", body or "")
    if match:
        return match.group(1)
    return None


def _issue_body(summary_markdown: str, digest: str) -> str:
    return f"{summary_markdown.rstrip()}\n\n{MARKER_PREFIX}{digest} -->\n"


def _comment_body(summary: Dict[str, Any]) -> str:
    next_task = summary.get("next_task", {})
    passed = sum(1 for row in summary.get("verification", []) if row.get("passed"))
    total = len(summary.get("verification", []))
    drift = summary.get("roadmap_drift", {}).get("signals", [])
    return "\n".join(
        [
            "Nightly audit changed.",
            "",
            f"- Status: `{summary.get('status', 'unknown')}`",
            f"- Verification: `{passed}/{total}` checks passed",
            f"- Drift signals: `{len(drift)}`",
            f"- Next best additive task: {next_task.get('title', 'n/a')}",
            f"- Classification: `{next_task.get('classification', 'unknown')}`",
            f"- Summary digest: `{summary.get('summary_digest', '')}`",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the nightly audit issue.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-markdown", required=True)
    parser.add_argument(
        "--issue-title",
        default="Economic World Model Nightly Audit",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        raise SystemExit("GITHUB_TOKEN and GITHUB_REPOSITORY are required")

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    summary_markdown = Path(args.summary_markdown).read_text(encoding="utf-8")
    digest = str(summary["summary_digest"])

    client = GitHubIssueClient(token=token, repository=repository)
    existing = None
    for issue in client.list_open_issues():
        if issue.get("title") == args.issue_title:
            existing = issue
            break

    body = _issue_body(summary_markdown, digest)
    if existing is None:
        client.create_issue(args.issue_title, body)
        return

    previous_digest = _extract_digest(existing.get("body", ""))
    if previous_digest == digest:
        return

    issue_number = int(existing["number"])
    client.update_issue(issue_number, body)
    client.add_comment(issue_number, _comment_body(summary))


if __name__ == "__main__":
    main()
