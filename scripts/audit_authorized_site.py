import argparse
import json
import sys

from cyberarena.site_audit import audit_site, save_report


def main():
    parser = argparse.ArgumentParser(
        description="CyberArena SiteGuard: passive authorized website assessment."
    )

    parser.add_argument("--url", required=True)
    parser.add_argument("--authorized-host", required=True)
    parser.add_argument(
        "--scope-confirmed",
        action="store_true",
        help="Confirm that you own or have written permission to assess this host.",
    )
    parser.add_argument(
        "--allow-private-lab",
        action="store_true",
        help="Allow private IPs for an explicitly authorized local lab.",
    )
    parser.add_argument(
        "--output",
        default="results/siteguard_report.json",
    )

    args = parser.parse_args()

    if not args.scope_confirmed:
        parser.error(
            "Confirm authorization using --scope-confirmed."
        )

    try:
        result = audit_site(
            args.url,
            args.authorized_host,
            allow_private_lab=args.allow_private_lab,
        )

        save_report(result, args.output)

        print("\nCYBERARENA SITEGUARD")
        print("=" * 50)
        print("Target:", result.target)
        print("Final URL:", result.final_url)
        print("HTTP status:", result.status_code)
        print("Findings:", len(result.findings))

        for finding in result.findings:
            print(
                f"[{finding.severity.upper()}] "
                f"{finding.check}: {finding.status}"
            )
            print(" ", finding.detail)
            print("  Recommendation:", finding.recommendation)

        print("\nJSON report:", args.output)

    except (ValueError, OSError, TimeoutError, Exception) as exc:
        print(f"Assessment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
