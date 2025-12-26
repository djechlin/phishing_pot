#!/usr/bin/env python3
"""Parse .eml files and extract HTML/text content as markdown."""

import sys
import email
from email import policy
from pathlib import Path
import html2text
import base64
import quopri
from typing import Optional, Tuple


def decode_payload(payload, encoding):
    """Decode payload based on content transfer encoding."""
    if encoding == "base64":
        try:
            return base64.b64decode(payload).decode("utf-8", errors="replace")
        except Exception:
            return payload.decode("utf-8", errors="replace")
    elif encoding == "quoted-printable":
        try:
            return quopri.decodestring(payload).decode("utf-8", errors="replace")
        except Exception:
            return payload.decode("utf-8", errors="replace")
    else:
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return payload


def extract_content(msg):
    """Extract HTML and text content from email message."""
    html_content = None
    text_content = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            encoding = part.get("Content-Transfer-Encoding", "").lower()

            if content_type == "text/html":
                payload = part.get_payload(decode=False)
                if isinstance(payload, bytes):
                    html_content = decode_payload(payload, encoding)
                else:
                    html_content = decode_payload(payload.encode() if payload else b"", encoding)
            elif content_type == "text/plain" and text_content is None:
                payload = part.get_payload(decode=False)
                if isinstance(payload, bytes):
                    text_content = decode_payload(payload, encoding)
                else:
                    text_content = decode_payload(payload.encode() if payload else b"", encoding)
    else:
        content_type = msg.get_content_type()
        encoding = msg.get("Content-Transfer-Encoding", "").lower()
        payload = msg.get_payload(decode=False)

        if isinstance(payload, bytes):
            decoded = decode_payload(payload, encoding)
        else:
            decoded = decode_payload(payload.encode() if payload else b"", encoding)

        if content_type == "text/html":
            html_content = decoded
        else:
            text_content = decoded

    return html_content, text_content


def html_to_markdown(html: str) -> str:
    """Convert HTML to markdown."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0  # Don't wrap lines
    return h.handle(html)


def parse_email_to_markdown(eml_path: Path) -> str:
    """Parse an .eml file and return markdown content."""
    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    # Extract headers we care about
    headers = []
    for header in ["From", "To", "Subject", "Date"]:
        value = msg.get(header, "")
        if value:
            headers.append(f"**{header}:** {value}")

    header_section = "\n".join(headers)

    # Extract content
    html_content, text_content = extract_content(msg)

    if html_content:
        body = html_to_markdown(html_content)
    elif text_content:
        body = text_content
    else:
        return None

    return f"# Email Content\n\n{header_section}\n\n---\n\n{body}"


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_email.py <email.eml> [email2.eml ...]", file=sys.stderr)
        sys.exit(1)

    failed = False
    for eml_file in sys.argv[1:]:
        eml_path = Path(eml_file)
        if not eml_path.exists():
            print(f"File not found: {eml_file}", file=sys.stderr)
            failed = True
            continue

        markdown = parse_email_to_markdown(eml_path)
        if markdown is None:
            print(f"No HTML or text content found in: {eml_file}", file=sys.stderr)
            failed = True
            continue

        output_path = eml_path.with_suffix(".md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Created: {output_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
