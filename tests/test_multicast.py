"""net/multicast.py against real sockets: TTL 0, so nothing leaves the host.

The case that matters: two publishers on one vision port. ssl-vision-client
keys frames by camera id and keeps the newest, so a second source does not
error -- it flickers the browser between two worlds. `senders()` is how an
app finds out before it starts publishing.
"""

import threading
import time

from tbots.net.multicast import senders, tx_socket

GROUP = "224.5.23.2"
PORT = 10599  # nothing in the stack or the league uses this


def test_senders_hears_a_publisher_already_on_the_port():
    tx = tx_socket()
    tx.sendto(b"x", (GROUP, PORT))   # binds the ephemeral source port
    src_port = tx.getsockname()[1]
    stop = threading.Event()

    def publish() -> None:
        while not stop.is_set():
            tx.sendto(b"x", (GROUP, PORT))
            time.sleep(0.01)

    t = threading.Thread(target=publish)
    t.start()
    try:
        heard = senders(GROUP, PORT, seconds=0.3)
    finally:
        stop.set()
        t.join()
        tx.close()

    assert {port for _, port in heard} == {src_port}


def test_senders_is_empty_on_a_quiet_port():
    assert senders(GROUP, PORT, seconds=0.2) == set()
