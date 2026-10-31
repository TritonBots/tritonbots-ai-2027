"""UDP multicast helpers.

Two things every new person needs to know:

1. League multicast packets are BARE PROTOBUF. One message per datagram,
   no length prefix. Just ParseFromString(data).
   The TCP interfaces (game-controller team client on 10008, CI on 10009)
   are length-delimited streams instead — different framing entirely.

2. In development, ALWAYS set multicast TTL to 0. TTL 0 means the packet
   never leaves this host. Without it, two people on the same lab wifi will
   silently referee each other's matches, and you will lose an afternoon.
"""

from __future__ import annotations

import select
import socket
import struct
import time


def rx_socket(group: str, port: int, iface: str = "0.0.0.0",
              blocking: bool = False) -> socket.socket:
    """Join a multicast group and return a socket ready to recvfrom()."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    s.bind(("", port))
    mreq = struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton(iface))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.setblocking(blocking)
    return s


def tx_socket(ttl: int = 0, iface: str = "0.0.0.0") -> socket.socket:
    """Socket for sending multicast. ttl=0 keeps packets on this host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                 socket.inet_aton(iface))
    return s


def senders(group: str, port: int, seconds: float = 0.5,
            iface: str = "0.0.0.0") -> set[tuple[str, int]]:
    """Every source address heard on group:port within `seconds`.

    Call this BEFORE you start publishing, so anything it hears is someone
    else. Two vision publishers on one port do not error: ssl-vision-client
    keeps the newest frame per camera id, so it silently flickers between
    the two worlds. Vision sources publish continuously (ER-Force at ~66 Hz,
    idle or not), so half a second is plenty to hear one.
    """
    s = rx_socket(group, port, iface)
    heard: set[tuple[str, int]] = set()
    deadline = time.monotonic() + seconds
    try:
        while (left := deadline - time.monotonic()) > 0:
            if select.select([s], [], [], left)[0]:
                try:
                    _, addr = s.recvfrom(65535)
                except (BlockingIOError, InterruptedError):
                    continue
                heard.add((addr[0], addr[1]))
    finally:
        s.close()
    return heard


def drain(sock: socket.socket, bufsize: int = 65535) -> list[bytes]:
    """Read every pending datagram without blocking. Newest is last.

    Call this once per control tick. Never block the control loop on a
    socket: one dropped packet would stall all six robots.
    """
    out: list[bytes] = []
    while True:
        try:
            out.append(sock.recv(bufsize))
        except (BlockingIOError, InterruptedError):
            return out
        except OSError:
            return out
