#!/usr/bin/env python3
"""
GitHub API helper for repository automation.

Usage:
    # Set token in environment
    export GITHUB_TOKEN='your_token_here'

    # Or pass directly
    api = GitHubAPI(token='your_token')

    # PR operations
    api.create_pr(head='feature-branch', base='main', title='...', body='...')
    api.get_pr_status(pr_number=1)
    api.merge_pr(pr_number=1, method='squash')

    # CI operations
    api.get_check_runs(sha='abc123')
    api.wait_for_ci(sha='abc123', timeout=600)

    # Release operations
    api.create_release(tag='v1.0.0', name='Release 1.0.0', body='...')
    api.list_releases()

    # Issue operations
    api.create_issue(title='...', body='...', labels=['bug'])
    api.update_issue(issue_number=1, state='closed')
    api.add_issue_comment(issue_number=1, body='...')

    # Branch protection
    api.get_branch_protection(branch='main')
    api.set_branch_protection(branch='main', required_checks=['test'])
"""

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GitHubConfig:
    owner: str = "robotics-vp"
    repo: str = "robotics-vp-core"
    token: Optional[str] = None

    def __post_init__(self):
        if self.token is None:
            self.token = os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not set in environment or passed directly")


class GitHubAPI:
    """GitHub API wrapper for repository automation."""

    def __init__(
        self,
        token: Optional[str] = None,
        owner: str = "robotics-vp",
        repo: str = "robotics-vp-core",
    ):
        self.config = GitHubConfig(owner=owner, repo=repo, token=token)
        self.base_url = f"https://api.github.com/repos/{self.config.owner}/{self.config.repo}"

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        full_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to the GitHub API."""
        url = full_url or f"{self.base_url}/{endpoint}"

        headers = {
            "Authorization": f"token {self.config.token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        }

        body = json.dumps(data).encode() if data else None

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            resp = urllib.request.urlopen(req)
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            raise RuntimeError(f"GitHub API error {e.code}: {error_body}")

    # ==========================================================================
    # Pull Request Operations
    # ==========================================================================

    def create_pr(
        self,
        head: str,
        base: str,
        title: str,
        body: str,
        draft: bool = False,
    ) -> Dict[str, Any]:
        """Create a pull request."""
        return self._request(
            "pulls",
            method="POST",
            data={
                "head": head,
                "base": base,
                "title": title,
                "body": body,
                "draft": draft,
            },
        )

    def get_pr(self, pr_number: int) -> Dict[str, Any]:
        """Get pull request details."""
        return self._request(f"pulls/{pr_number}")

    def get_pr_status(self, pr_number: int) -> Dict[str, Any]:
        """Get PR with CI status information."""
        pr = self.get_pr(pr_number)
        sha = pr["head"]["sha"]
        checks = self.get_check_runs(sha)
        return {
            "pr": pr,
            "checks": checks,
            "mergeable": pr.get("mergeable"),
            "state": pr["state"],
        }

    def merge_pr(
        self,
        pr_number: int,
        method: str = "squash",
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge a pull request."""
        data = {"merge_method": method}
        if commit_title:
            data["commit_title"] = commit_title
        if commit_message:
            data["commit_message"] = commit_message
        return self._request(f"pulls/{pr_number}/merge", method="PUT", data=data)

    def list_prs(
        self,
        state: str = "open",
        base: Optional[str] = None,
        head: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List pull requests."""
        params = [f"state={state}"]
        if base:
            params.append(f"base={base}")
        if head:
            params.append(f"head={head}")
        query = "&".join(params)
        return self._request(f"pulls?{query}")

    def add_pr_comment(self, pr_number: int, body: str) -> Dict[str, Any]:
        """Add a comment to a pull request."""
        return self._request(
            f"issues/{pr_number}/comments", method="POST", data={"body": body}
        )

    # ==========================================================================
    # CI / Check Runs Operations
    # ==========================================================================

    def get_check_runs(self, sha: str) -> Dict[str, Any]:
        """Get check runs for a commit."""
        return self._request(f"commits/{sha}/check-runs")

    def get_commit_status(self, sha: str) -> Dict[str, Any]:
        """Get combined status for a commit."""
        return self._request(f"commits/{sha}/status")

    def wait_for_ci(
        self,
        sha: str,
        timeout: int = 600,
        poll_interval: int = 30,
    ) -> Dict[str, Any]:
        """Wait for CI to complete on a commit."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            checks = self.get_check_runs(sha)

            if checks["total_count"] == 0:
                time.sleep(poll_interval)
                continue

            all_complete = True
            all_success = True
            results = []

            for cr in checks.get("check_runs", []):
                status = cr["status"]
                conclusion = cr.get("conclusion")
                results.append(
                    {"name": cr["name"], "status": status, "conclusion": conclusion}
                )

                if status != "completed":
                    all_complete = False
                if conclusion not in ["success", "skipped", None]:
                    all_success = False

            if all_complete:
                return {
                    "success": all_success,
                    "results": results,
                    "elapsed": time.time() - start_time,
                }

            time.sleep(poll_interval)

        raise TimeoutError(f"CI did not complete within {timeout} seconds")

    # ==========================================================================
    # Release Operations
    # ==========================================================================

    def create_release(
        self,
        tag: str,
        name: str,
        body: str,
        target_commitish: str = "main",
        draft: bool = False,
        prerelease: bool = False,
    ) -> Dict[str, Any]:
        """Create a new release."""
        return self._request(
            "releases",
            method="POST",
            data={
                "tag_name": tag,
                "target_commitish": target_commitish,
                "name": name,
                "body": body,
                "draft": draft,
                "prerelease": prerelease,
            },
        )

    def list_releases(self, per_page: int = 10) -> List[Dict[str, Any]]:
        """List releases."""
        return self._request(f"releases?per_page={per_page}")

    def get_release(self, release_id: int) -> Dict[str, Any]:
        """Get a specific release."""
        return self._request(f"releases/{release_id}")

    def get_latest_release(self) -> Dict[str, Any]:
        """Get the latest release."""
        return self._request("releases/latest")

    def delete_release(self, release_id: int) -> None:
        """Delete a release."""
        self._request(f"releases/{release_id}", method="DELETE")

    # ==========================================================================
    # Issue Operations
    # ==========================================================================

    def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        milestone: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an issue."""
        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        if milestone:
            data["milestone"] = milestone
        return self._request("issues", method="POST", data=data)

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """Get an issue."""
        return self._request(f"issues/{issue_number}")

    def update_issue(
        self,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update an issue."""
        data = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        if state:
            data["state"] = state
        if labels is not None:
            data["labels"] = labels
        if assignees is not None:
            data["assignees"] = assignees
        return self._request(f"issues/{issue_number}", method="PATCH", data=data)

    def add_issue_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        """Add a comment to an issue."""
        return self._request(
            f"issues/{issue_number}/comments", method="POST", data={"body": body}
        )

    def list_issues(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        per_page: int = 30,
    ) -> List[Dict[str, Any]]:
        """List issues."""
        params = [f"state={state}", f"per_page={per_page}"]
        if labels:
            params.append(f"labels={','.join(labels)}")
        query = "&".join(params)
        return self._request(f"issues?{query}")

    # ==========================================================================
    # Branch Protection Operations
    # ==========================================================================

    def get_branch_protection(self, branch: str = "main") -> Dict[str, Any]:
        """Get branch protection settings."""
        try:
            return self._request(f"branches/{branch}/protection")
        except RuntimeError as e:
            if "404" in str(e):
                return {"enabled": False}
            raise

    def set_branch_protection(
        self,
        branch: str = "main",
        required_checks: Optional[List[str]] = None,
        strict: bool = True,
        required_reviews: int = 1,
        dismiss_stale_reviews: bool = True,
        enforce_admins: bool = False,
        allow_force_pushes: bool = False,
        allow_deletions: bool = False,
    ) -> Dict[str, Any]:
        """Set branch protection rules."""
        data = {
            "required_status_checks": {
                "strict": strict,
                "contexts": required_checks or ["test"],
            },
            "enforce_admins": enforce_admins,
            "required_pull_request_reviews": {
                "required_approving_review_count": required_reviews,
                "dismiss_stale_reviews": dismiss_stale_reviews,
                "require_code_owner_reviews": False,
            },
            "restrictions": None,
            "required_linear_history": False,
            "allow_force_pushes": allow_force_pushes,
            "allow_deletions": allow_deletions,
        }
        return self._request(f"branches/{branch}/protection", method="PUT", data=data)

    def remove_branch_protection(self, branch: str = "main") -> None:
        """Remove branch protection."""
        self._request(f"branches/{branch}/protection", method="DELETE")

    # ==========================================================================
    # Repository Operations
    # ==========================================================================

    def get_repo(self) -> Dict[str, Any]:
        """Get repository information."""
        url = f"https://api.github.com/repos/{self.config.owner}/{self.config.repo}"
        return self._request("", full_url=url)

    def list_branches(self, per_page: int = 30) -> List[Dict[str, Any]]:
        """List branches."""
        return self._request(f"branches?per_page={per_page}")

    def get_branch(self, branch: str) -> Dict[str, Any]:
        """Get branch information."""
        return self._request(f"branches/{branch}")

    def list_tags(self, per_page: int = 30) -> List[Dict[str, Any]]:
        """List tags."""
        return self._request(f"tags?per_page={per_page}")

    # ==========================================================================
    # Workflow Operations
    # ==========================================================================

    def list_workflows(self) -> Dict[str, Any]:
        """List workflows."""
        return self._request("actions/workflows")

    def get_workflow_runs(
        self,
        workflow_id: str,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """Get workflow runs."""
        return self._request(
            f"actions/workflows/{workflow_id}/runs?per_page={per_page}"
        )

    def trigger_workflow(
        self,
        workflow_id: str,
        ref: str = "main",
        inputs: Optional[Dict[str, str]] = None,
    ) -> None:
        """Trigger a workflow dispatch event."""
        data = {"ref": ref}
        if inputs:
            data["inputs"] = inputs
        self._request(
            f"actions/workflows/{workflow_id}/dispatches", method="POST", data=data
        )


# ==========================================================================
# CLI Interface
# ==========================================================================

def main():
    """CLI entry point for common operations."""
    import argparse

    parser = argparse.ArgumentParser(description="GitHub API CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # PR commands
    pr_parser = subparsers.add_parser("pr", help="PR operations")
    pr_sub = pr_parser.add_subparsers(dest="pr_command")

    pr_list = pr_sub.add_parser("list", help="List PRs")
    pr_list.add_argument("--state", default="open", help="PR state")

    pr_status = pr_sub.add_parser("status", help="Get PR status")
    pr_status.add_argument("number", type=int, help="PR number")

    pr_merge = pr_sub.add_parser("merge", help="Merge PR")
    pr_merge.add_argument("number", type=int, help="PR number")
    pr_merge.add_argument("--method", default="squash", help="Merge method")

    # CI commands
    ci_parser = subparsers.add_parser("ci", help="CI operations")
    ci_sub = ci_parser.add_subparsers(dest="ci_command")

    ci_status = ci_sub.add_parser("status", help="Get CI status")
    ci_status.add_argument("sha", help="Commit SHA")

    ci_wait = ci_sub.add_parser("wait", help="Wait for CI")
    ci_wait.add_argument("sha", help="Commit SHA")
    ci_wait.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")

    # Protection commands
    protect_parser = subparsers.add_parser("protect", help="Branch protection")
    protect_sub = protect_parser.add_subparsers(dest="protect_command")

    protect_get = protect_sub.add_parser("get", help="Get protection")
    protect_get.add_argument("--branch", default="main", help="Branch name")

    protect_set = protect_sub.add_parser("set", help="Set protection")
    protect_set.add_argument("--branch", default="main", help="Branch name")
    protect_set.add_argument("--checks", nargs="+", default=["test"], help="Required checks")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    api = GitHubAPI()

    if args.command == "pr":
        if args.pr_command == "list":
            prs = api.list_prs(state=args.state)
            for pr in prs:
                print(f"#{pr['number']}: {pr['title']} ({pr['state']})")
        elif args.pr_command == "status":
            status = api.get_pr_status(args.number)
            print(f"PR #{args.number}: {status['pr']['title']}")
            print(f"State: {status['state']}")
            print(f"Mergeable: {status['mergeable']}")
            for check in status["checks"].get("check_runs", []):
                print(f"  - {check['name']}: {check['status']} / {check.get('conclusion')}")
        elif args.pr_command == "merge":
            result = api.merge_pr(args.number, method=args.method)
            print(f"Merged: {result.get('sha')}")

    elif args.command == "ci":
        if args.ci_command == "status":
            checks = api.get_check_runs(args.sha)
            print(f"Total checks: {checks['total_count']}")
            for check in checks.get("check_runs", []):
                print(f"  - {check['name']}: {check['status']} / {check.get('conclusion')}")
        elif args.ci_command == "wait":
            result = api.wait_for_ci(args.sha, timeout=args.timeout)
            print(f"CI {'passed' if result['success'] else 'failed'} in {result['elapsed']:.1f}s")

    elif args.command == "protect":
        if args.protect_command == "get":
            protection = api.get_branch_protection(args.branch)
            print(json.dumps(protection, indent=2))
        elif args.protect_command == "set":
            result = api.set_branch_protection(args.branch, required_checks=args.checks)
            print(f"Protection set on {args.branch}")
            print(f"Required checks: {args.checks}")


if __name__ == "__main__":
    main()
