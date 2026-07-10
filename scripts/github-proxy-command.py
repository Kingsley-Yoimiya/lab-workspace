#!/usr/bin/env python3
"""HTTP CONNECT ProxyCommand for OpenSSH.
Usage: github-proxy-command.py <host> <port>
Reads proxy from PASEO_EGRESS_PROXY / HTTPS_PROXY / http://127.0.0.1:32262
"""
import os
import socket
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} host port", file=sys.stderr)
        sys.exit(2)
    host, port = sys.argv[1], sys.argv[2]
    proxy = (
        os.environ.get("PASEO_EGRESS_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
        or "http://127.0.0.1:32262"
    )
    # strip scheme
    if "://" in proxy:
        proxy = proxy.split("://", 1)[1]
    # strip userinfo if any
    if "@" in proxy:
        proxy = proxy.rsplit("@", 1)[-1]
    ph, pp = proxy.rsplit(":", 1)

    s = socket.create_connection((ph, int(pp)), timeout=30)
    req = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode()
    s.sendall(req)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    status = data.split(b"\r\n", 1)[0]
    if b" 200 " not in status:
        sys.stderr.write(f"proxy CONNECT failed: {status!r}\n")
        sys.exit(1)

    # hand socket to ssh via stdin/stdout
    try:
        import selectors

        sel = selectors.DefaultSelector()
        sel.register(s, selectors.EVENT_READ)
        sel.register(sys.stdin.buffer, selectors.EVENT_READ)
        while True:
            for key, _ in sel.select():
                if key.fileobj is s:
                    buf = s.recv(65536)
                    if not buf:
                        return
                    sys.stdout.buffer.write(buf)
                    sys.stdout.buffer.flush()
                else:
                    buf = sys.stdin.buffer.read1(65536)
                    if not buf:
                        return
                    s.sendall(buf)
    finally:
        s.close()


if __name__ == "__main__":
    main()
