#!/usr/bin/env python3
"""CLI for valuation verification.

Verifies provenance closure and SHA consistency for a run.

Usage:
    python scripts/verify_valuation.py --run-dir artifacts/hardened
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Verify valuation provenance for a run"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run output directory (containing run_manifest.json, ledger.jsonl, etc.)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write verification report JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, not detailed checks",
    )
    args = parser.parse_args()

    # Import here to avoid circular imports
    from src.valuation.valuation_verifier import verify_run, write_verification_report

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Verifying run at: {run_dir}")
    print("-" * 60)

    report = verify_run(str(run_dir))

    # Print summary
    status = "✅ PASSED" if report.all_passed else "❌ FAILED"
    print(f"\nRun ID: {report.run_id}")
    print(f"Status: {status}")
    print(f"Checks: {report.passed_count}/{report.check_count} passed")
    print(f"Ledger records: {report.ledger_record_count}")
    if report.manifest_sha:
        print(f"Manifest SHA: {report.manifest_sha[:16]}...")

    # Print warnings
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for warn in report.warnings:
            print(f"  ⚠️  {warn}")

    # Print failed checks
    failed = [c for c in report.checks if not c.passed]
    if failed and not args.quiet:
        print(f"\nFailed checks ({len(failed)}):")
        for check in failed:
            print(f"  ✗ {check.check_id}: {check.message}")
            if check.expected:
                print(f"      expected: {check.expected}")
            if check.actual:
                print(f"      actual:   {check.actual}")

    # Print all checks if verbose
    if not args.quiet and report.all_passed:
        print(f"\nAll checks ({report.check_count}):")
        for check in report.checks:
            symbol = "✓" if check.passed else "✗"
            print(f"  {symbol} {check.check_id}")

    # Write report if requested
    if args.output:
        report_sha = write_verification_report(args.output, report)
        print(f"\nVerification report written to: {args.output}")
        print(f"Report SHA: {report_sha[:16]}...")

    print("-" * 60)
    print(f"Verification SHA: {report.sha256()[:16]}...")

    # Exit with appropriate code
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
