from unittest.mock import patch

import pytest

import cyberarena.site_audit as site_audit


PUBLIC_DNS = [
    (2, 1, 6, "", ("93.184.216.34", 443))
]


def mock_dns(*args, **kwargs):
    return PUBLIC_DNS


def test_authorized_hostname_must_match():
    with pytest.raises(ValueError, match="does not match"):
        site_audit.validate_target(
            "https://example.com",
            "another.example",
        )


def test_private_ip_is_blocked_by_default():
    private_dns = [
        (2, 1, 6, "", ("127.0.0.1", 443))
    ]

    with patch.object(site_audit.socket, "getaddrinfo", return_value=private_dns):
        with pytest.raises(ValueError, match="Private/local IP blocked"):
            site_audit.validate_target(
                "https://example.test",
                "example.test",
            )


def test_nonstandard_port_is_rejected():
    with pytest.raises(ValueError, match="standard HTTP/HTTPS ports"):
        site_audit.validate_target(
            "https://example.com:8443",
            "example.com",
        )


def test_cross_host_redirect_is_blocked():
    with patch.object(site_audit.socket, "getaddrinfo", side_effect=mock_dns):
        with patch.object(
            site_audit,
            "request_headers",
            return_value=(
                302,
                [("Location", "https://evil.example/path")],
                "https://evil.example/path",
                "TLSv1.3",
            ),
        ):
            with pytest.raises(ValueError, match="Cross-host redirect"):
                site_audit.audit_site(
                    "https://example.com",
                    "example.com",
                )


def test_https_downgrade_redirect_is_blocked():
    with patch.object(site_audit.socket, "getaddrinfo", side_effect=mock_dns):
        with patch.object(
            site_audit,
            "request_headers",
            return_value=(
                302,
                [("Location", "http://example.com/path")],
                "http://example.com/path",
                "TLSv1.3",
            ),
        ):
            with pytest.raises(ValueError, match="downgrade"):
                site_audit.audit_site(
                    "https://example.com",
                    "example.com",
                )


def test_missing_security_headers_are_reported():
    headers = [
        ("Content-Type", "text/html"),
        ("Server", "Example"),
    ]

    with patch.object(site_audit.socket, "getaddrinfo", side_effect=mock_dns):
        with patch.object(
            site_audit,
            "request_headers",
            return_value=(200, headers, None, "TLSv1.3"),
        ):
            result = site_audit.audit_site(
                "https://example.com",
                "example.com",
            )

    names = {finding.check for finding in result.findings}
    assert "HSTS" in names
    assert "Content Security Policy" in names
    assert "Clickjacking protection" in names

