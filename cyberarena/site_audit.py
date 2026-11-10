from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import ssl

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit


USER_AGENT = "CyberArena-SiteGuard/1.0"
TIMEOUT = 8
MAX_REDIRECTS = 3


@dataclass
class Finding:
    check: str
    severity: str
    status: str
    detail: str
    recommendation: str


@dataclass
class AuditResult:
    target: str
    final_url: str
    status_code: int | None
    timestamp: str
    findings: list[Finding]

    def to_dict(self):
        return {
            "target": self.target,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "timestamp": self.timestamp,
            "findings": [asdict(item) for item in self.findings],
        }


def validate_target(url: str, authorized_host: str, allow_private_lab=False):
    parsed = urlsplit(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not parsed.hostname:
        raise ValueError("A valid hostname is required.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    host = parsed.hostname.lower().rstrip(".")
    approved = authorized_host.lower().strip().rstrip(".")

    if host != approved:
        raise ValueError(
            "Target hostname does not match --authorized-host."
        )

    if parsed.port not in (None, 80, 443):
        raise ValueError("Only standard HTTP/HTTPS ports are allowed.")

    if parsed.scheme == "https" and parsed.port not in (None, 443):
        raise ValueError("HTTPS must use the standard port.")

    if parsed.scheme == "http" and parsed.port not in (None, 80):
        raise ValueError("HTTP must use the standard port.")

    try:
        records = socket.getaddrinfo(
            host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed: {exc}") from exc

    addresses = sorted({record[4][0] for record in records})

    if not addresses:
        raise ValueError("Hostname did not resolve to an IP address.")

    for address in addresses:
        ip = ipaddress.ip_address(address)

        if ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            raise ValueError("Target resolves to a restricted IP address.")

        if not allow_private_lab and not ip.is_global:
            raise ValueError(
                "Private/local IP blocked. Use the explicit lab option "
                "only for a system you own."
            )

        if allow_private_lab and ip.is_link_local:
            raise ValueError("Link-local targets are not supported.")

    return parsed, addresses[0]


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a resolved IP while preserving hostname-based TLS SNI."""

    def __init__(self, ip, hostname, port, timeout):
        super().__init__(
            ip,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self.original_hostname = hostname

    def connect(self):
        self.sock = socket.create_connection(
            (self.host, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname=self.original_hostname,
        )


def request_headers(url, allow_private_lab=False):
    parsed = urlsplit(url)
    host = parsed.hostname.lower().rstrip(".")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    _, ip = validate_target(url, host, allow_private_lab)

    if parsed.scheme == "https":
        connection = PinnedHTTPSConnection(
            ip, host, port, TIMEOUT
        )
    else:
        connection = http.client.HTTPConnection(
            ip, port=port, timeout=TIMEOUT
        )

    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    host_header = host
    if ":" in host:
        host_header = f"[{host}]"

    headers = {
        "Host": host_header,
        "User-Agent": USER_AGENT,
        "Connection": "close",
    }

    try:
        connection.request("HEAD", path, headers=headers)
        response = connection.getresponse()

        response_headers = response.getheaders()
        status = response.status
        location = response.getheader("Location")

        tls_version = None
        if parsed.scheme == "https" and connection.sock:
            tls_version = connection.sock.version()

        return status, response_headers, location, tls_version

    finally:
        connection.close()


def add(findings, check, severity, status, detail, recommendation):
    findings.append(
        Finding(check, severity, status, detail, recommendation)
    )


def audit_site(url, authorized_host, allow_private_lab=False):
    parsed, _ = validate_target(
        url, authorized_host, allow_private_lab
    )

    original_host = parsed.hostname.lower().rstrip(".")
    current_url = url
    findings = []
    final_status = None
    final_headers = []
    tls_version = None

    for redirect_count in range(MAX_REDIRECTS + 1):
        parsed_current = urlsplit(current_url)

        if parsed_current.hostname.lower().rstrip(".") != original_host:
            raise ValueError("Cross-host redirect blocked.")

        if parsed_current.port not in (None, 80, 443):
            raise ValueError("Redirect to non-standard port blocked.")

        if parsed.scheme == "https" and parsed_current.scheme != "https":
            raise ValueError("HTTPS downgrade redirect blocked.")

        if parsed_current.scheme not in ("http", "https"):
            raise ValueError("Unsupported redirect scheme.")

        status, headers, location, tls_version = request_headers(
            current_url, allow_private_lab
        )

        final_status = status
        final_headers = headers

        if status not in (301, 302, 303, 307, 308) or not location:
            break

        if redirect_count == MAX_REDIRECTS:
            raise ValueError("Maximum redirect count exceeded.")

        next_url = urljoin(current_url, location)
        next_parsed = urlsplit(next_url)

        if next_parsed.hostname is None:
            raise ValueError("Invalid redirect destination.")

        if next_parsed.hostname.lower().rstrip(".") != original_host:
            raise ValueError("Cross-host redirect blocked.")

        if parsed.scheme == "https" and next_parsed.scheme != "https":
            raise ValueError("HTTPS downgrade redirect blocked.")

        current_url = next_url
    else:
        raise ValueError("Redirect processing failed.")

    headers_map = {}
    cookies = []

    for key, value in final_headers:
        key_lower = key.lower()

        if key_lower == "set-cookie":
            cookies.append(value)
        else:
            headers_map[key_lower] = value

    if parsed.scheme == "https":
        add(
            findings, "TLS certificate", "info", "pass",
            f"TLS connection and certificate verification succeeded. "
            f"Protocol: {tls_version or 'not reported'}.",
            "Continue monitoring certificate expiry and TLS configuration.",
        )
    elif urlsplit(current_url).scheme == "https":
        add(
            findings, "HTTPS redirect", "info", "pass",
            "The original HTTP URL redirected to HTTPS.",
            "Keep HTTP-to-HTTPS redirection enabled.",
        )
    else:
        add(
            findings, "HTTPS", "high", "fail",
            "The target did not redirect the HTTP request to HTTPS.",
            "Enable HTTPS and redirect HTTP requests to HTTPS.",
        )

    if urlsplit(current_url).scheme == "https":
        checks = [
            (
                "strict-transport-security",
                "HSTS",
                "medium",
                "Enable Strict-Transport-Security after confirming "
                "all required subdomains support HTTPS.",
            ),
            (
                "content-security-policy",
                "Content Security Policy",
                "medium",
                "Define a restrictive Content-Security-Policy "
                "appropriate for the application.",
            ),
            (
                "x-content-type-options",
                "Content type protection",
                "low",
                "Set X-Content-Type-Options: nosniff.",
            ),
            (
                "referrer-policy",
                "Referrer policy",
                "low",
                "Set an explicit Referrer-Policy.",
            ),
            (
                "permissions-policy",
                "Permissions policy",
                "low",
                "Restrict browser features that the application "
                "does not require.",
            ),
        ]

        for header, name, severity, recommendation in checks:
            if header in headers_map:
                add(
                    findings, name, "info", "pass",
                    f"{name} header is present.",
                    "Review its value for application-specific correctness.",
                )
            else:
                add(
                    findings, name, severity, "warning",
                    f"{name} header is missing.",
                    recommendation,
                )

        csp = headers_map.get("content-security-policy", "")
        xfo = headers_map.get("x-frame-options", "")

        if "frame-ancestors" not in csp.lower() and not xfo:
            add(
                findings, "Clickjacking protection", "medium", "warning",
                "Neither CSP frame-ancestors nor X-Frame-Options was observed.",
                "Configure CSP frame-ancestors or X-Frame-Options.",
            )
        else:
            add(
                findings, "Clickjacking protection", "info", "pass",
                "A frame restriction header was observed.",
                "Verify the policy matches the application's embedding requirements.",
            )

    if cookies:
        for cookie in cookies:
            parts = [part.strip() for part in cookie.split(";")]
            cookie_name = parts[0].split("=", 1)[0][:80]
            flags = {
                part.split("=", 1)[0].strip().lower()
                for part in parts[1:]
            }

            missing = []
            if "secure" not in flags:
                missing.append("Secure")
            if "httponly" not in flags:
                missing.append("HttpOnly")
            if not any(flag.startswith("samesite") for flag in flags):
                missing.append("SameSite")

            if missing:
                add(
                    findings,
                    f"Cookie flags: {cookie_name}",
                    "medium",
                    "warning",
                    "Cookie is missing: " + ", ".join(missing),
                    "Set appropriate Secure, HttpOnly and SameSite attributes.",
                )
            else:
                add(
                    findings,
                    f"Cookie flags: {cookie_name}",
                    "info",
                    "pass",
                    "Secure, HttpOnly and SameSite attributes were observed.",
                    "Review cookie scope and lifetime as well.",
                )

    server = headers_map.get("server")
    if server:
        add(
            findings, "Server banner", "info", "warning",
            "The response exposes a Server header.",
            "Consider minimizing unnecessary server/version disclosure.",
        )

    if not cookies:
        add(
            findings, "Cookie inspection", "info", "not_observed",
            "No Set-Cookie header was returned by the HEAD request.",
            "This does not establish that the application never uses cookies.",
        )

    return AuditResult(
        target=url,
        final_url=current_url,
        status_code=final_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        findings=findings,
    )


def save_report(result, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return path
