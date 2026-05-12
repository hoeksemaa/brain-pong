"""
Direct Cerelog USB serial monitor.

This intentionally bypasses BrainFlow. It opens the serial device, optionally
sends the Cerelog handshake, then shows raw bytes and/or decoded 37-byte EEG
packets from the firmware.

Useful commands:
    source .venv/bin/activate
    python cerelog_serial_monitor.py
    python cerelog_serial_monitor.py --mode both --seconds 10
    python cerelog_serial_monitor.py --no-handshake --baud 115200 --mode hex
    python cerelog_serial_monitor.py --save-raw diagnostics/serial.raw --save-packets diagnostics/packets.csv
"""

import argparse
import csv
import glob
import os
import select
import struct
import sys
import termios
import time
import tty
from pathlib import Path


DEFAULT_PORT = "/dev/cu.usbserial-1120"
INITIAL_BAUD = 9600
FINAL_BAUD = 115200
FIRMWARE_BAUD_RATE_INDEX = 0x04

PACKET_START = b"\xAB\xCD"
PACKET_END = b"\xDC\xBA"
PACKET_SIZE = 37
CHECKSUM_INDEX = 34
ADS_OFFSET = 7
ADS_STATUS_BYTES = 3
ADS_CHANNELS = 8
ADS_BYTES_PER_CHANNEL = 3

VREF = 4.5
GAIN = 24.0
ADS_SCALE_UV = ((2.0 * VREF / GAIN) / (2**24)) * 1_000_000.0


def baud_constant(baud):
    name = f"B{baud}"
    if not hasattr(termios, name):
        supported = sorted(
            int(n[1:]) for n in dir(termios)
            if n.startswith("B") and n[1:].isdigit()
        )
        raise ValueError(f"baud {baud} is not exposed by termios; supported includes {supported}")
    return getattr(termios, name)


def configure_serial(fd, baud):
    attrs = termios.tcgetattr(fd)
    tty.setraw(fd)
    attrs = termios.tcgetattr(fd)

    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attrs
    iflag &= ~(termios.IXON | termios.IXOFF | termios.IXANY)
    oflag = 0
    lflag = 0
    cflag |= termios.CLOCAL | termios.CREAD
    cflag &= ~termios.PARENB
    cflag &= ~termios.CSTOPB
    cflag &= ~termios.CSIZE
    cflag |= termios.CS8
    if hasattr(termios, "CRTSCTS"):
        cflag &= ~termios.CRTSCTS
    if hasattr(termios, "CCTS_OFLOW"):
        cflag &= ~termios.CCTS_OFLOW
    if hasattr(termios, "CRTS_IFLOW"):
        cflag &= ~termios.CRTS_IFLOW

    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 1
    speed = baud_constant(baud)
    termios.tcsetattr(fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, speed, speed, cc])
    termios.tcflush(fd, termios.TCIOFLUSH)


def open_serial(port, baud):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure_serial(fd, baud)
    return fd


def read_available(fd, timeout_s=0.1, max_bytes=4096):
    ready, _, _ = select.select([fd], [], [], timeout_s)
    if not ready:
        return b""
    try:
        return os.read(fd, max_bytes)
    except BlockingIOError:
        return b""


def write_all(fd, data):
    offset = 0
    while offset < len(data):
        _, ready, _ = select.select([], [fd], [], 1.0)
        if not ready:
            raise TimeoutError("serial write timed out")
        offset += os.write(fd, data[offset:])


def build_handshake():
    unix_time = int(time.time())
    payload = struct.pack(">BI", 0x02, unix_time) + bytes([0x01, FIRMWARE_BAUD_RATE_INDEX])
    checksum = sum(payload) & 0xFF
    return b"\xAA\xBB" + payload + bytes([checksum]) + b"\xCC\xDD"


def handshake_and_reopen(port, initial_baud, final_baud, boot_wait_s, switch_wait_s):
    print(f"opening {port} at {initial_baud} baud for handshake")
    fd = open_serial(port, initial_baud)
    try:
        print(f"waiting {boot_wait_s:.1f}s for board/USB serial settle")
        time.sleep(boot_wait_s)
        read_available(fd, timeout_s=0.1, max_bytes=65536)

        packet = build_handshake()
        print("sending handshake: " + packet.hex(" "))
        write_all(fd, packet)
        time.sleep(switch_wait_s)
    finally:
        os.close(fd)

    print(f"reopening {port} at {final_baud} baud")
    time.sleep(0.2)
    fd = open_serial(port, final_baud)
    time.sleep(0.5)
    read_available(fd, timeout_s=0.1, max_bytes=65536)
    return fd


def printable_ascii(chunk):
    return "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)


def print_hex_chunk(chunk, max_len):
    view = chunk[:max_len]
    suffix = "" if len(chunk) <= max_len else f" ... +{len(chunk) - max_len} bytes"
    print(f"[hex] {len(chunk):5d} bytes  {view.hex(' ')}{suffix}")
    print(f"[asc]              {printable_ascii(view)}")


def parse_signed24(data):
    value = int.from_bytes(data, byteorder="big", signed=False)
    if value & 0x800000:
        value -= 0x1000000
    return value


def decode_packet(packet):
    checksum = sum(packet[2:CHECKSUM_INDEX]) & 0xFF
    checksum_ok = checksum == packet[CHECKSUM_INDEX]
    end_ok = packet[-2:] == PACKET_END
    board_ms = int.from_bytes(packet[3:7], byteorder="big", signed=False)
    ads = packet[ADS_OFFSET:CHECKSUM_INDEX]
    status = ads[:ADS_STATUS_BYTES]
    counts = []
    for ch in range(ADS_CHANNELS):
        idx = ADS_STATUS_BYTES + ch * ADS_BYTES_PER_CHANNEL
        counts.append(parse_signed24(ads[idx:idx + ADS_BYTES_PER_CHANNEL]))
    microvolts = [count * ADS_SCALE_UV for count in counts]
    return {
        "checksum_ok": checksum_ok,
        "end_ok": end_ok,
        "board_ms": board_ms,
        "status_hex": status.hex(" "),
        "counts": counts,
        "microvolts": microvolts,
    }


def extract_packets(buffer):
    packets = []
    dropped = 0
    while True:
        start = buffer.find(PACKET_START)
        if start < 0:
            if len(buffer) > PACKET_SIZE:
                dropped += len(buffer) - (PACKET_SIZE - 1)
                del buffer[:len(buffer) - (PACKET_SIZE - 1)]
            break
        if start > 0:
            dropped += start
            del buffer[:start]
        if len(buffer) < PACKET_SIZE:
            break
        packet = bytes(buffer[:PACKET_SIZE])
        decoded = decode_packet(packet)
        if decoded["end_ok"]:
            packets.append((packet, decoded))
            del buffer[:PACKET_SIZE]
        else:
            dropped += 1
            del buffer[:1]
    return packets, dropped


def format_packet_line(packet_index, decoded):
    uv = decoded["microvolts"]
    uv_text = " ".join(f"{v:9.1f}" for v in uv)
    counts = decoded["counts"]
    zeroish = sum(1 for value in counts if value == 0)
    flags = []
    if not decoded["checksum_ok"]:
        flags.append("BAD_CHECKSUM")
    if not decoded["end_ok"]:
        flags.append("BAD_END")
    if zeroish == ADS_CHANNELS:
        flags.append("ALL_ZERO_COUNTS")
    flag_text = ",".join(flags) if flags else "ok"
    return (
        f"[pkt {packet_index:06d}] ms={decoded['board_ms']:>10} "
        f"status={decoded['status_hex']:<8} {flag_text:<16} "
        f"uV={uv_text}"
    )


def list_candidate_ports():
    patterns = [
        "/dev/cu.usbserial*",
        "/dev/tty.usbserial*",
        "/dev/cu.wchusbserial*",
        "/dev/tty.wchusbserial*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ]
    ports = []
    for pattern in patterns:
        ports.extend(glob.glob(pattern))
    return sorted(set(ports))


def open_csv(path):
    if not path:
        return None, None
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", newline="")
    writer = csv.writer(handle)
    writer.writerow([
        "host_time_s",
        "packet_index",
        "board_ms",
        "checksum_ok",
        "end_ok",
        "status_hex",
        *[f"ch{i + 1}_count" for i in range(ADS_CHANNELS)],
        *[f"ch{i + 1}_uv" for i in range(ADS_CHANNELS)],
    ])
    return handle, writer


def main():
    parser = argparse.ArgumentParser(description="Direct raw USB serial monitor for Cerelog X8")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--baud", type=int, default=FINAL_BAUD)
    parser.add_argument("--initial-baud", type=int, default=INITIAL_BAUD)
    parser.add_argument("--handshake", dest="handshake", action="store_true", default=True)
    parser.add_argument("--no-handshake", dest="handshake", action="store_false")
    parser.add_argument("--boot-wait-s", type=float, default=5.0)
    parser.add_argument("--switch-wait-s", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means run until Ctrl-C")
    parser.add_argument("--mode", choices=["hex", "packets", "both"], default="packets")
    parser.add_argument("--hex-bytes", type=int, default=96)
    parser.add_argument("--packet-print-every", type=int, default=25)
    parser.add_argument("--stats-every-s", type=float, default=1.0)
    parser.add_argument("--save-raw", default=None)
    parser.add_argument("--save-packets", default=None)
    args = parser.parse_args()

    if args.list_ports:
        ports = list_candidate_ports()
        if ports:
            print("\n".join(ports))
        else:
            print("No likely serial ports found.")
        return

    raw_handle = None
    csv_handle = None
    csv_writer = None
    fd = None
    try:
        if args.save_raw:
            raw_path = Path(args.save_raw)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_handle = raw_path.open("ab")
        csv_handle, csv_writer = open_csv(args.save_packets)

        if args.handshake:
            fd = handshake_and_reopen(
                args.port,
                args.initial_baud,
                args.baud,
                args.boot_wait_s,
                args.switch_wait_s,
            )
        else:
            print(f"opening {args.port} at {args.baud} baud without handshake")
            fd = open_serial(args.port, args.baud)

        print("monitoring; Ctrl-C to stop")
        print("packet uV scale: ADS1299 counts * ((2 * 4.5V / 24) / 2^24) * 1e6")

        buffer = bytearray()
        packet_index = 0
        bytes_seen = 0
        bytes_last = 0
        packets_last = 0
        dropped_total = 0
        bad_checksum_total = 0
        all_zero_total = 0
        start_t = time.monotonic()
        last_stats_t = start_t

        while True:
            now = time.monotonic()
            if args.seconds > 0 and now - start_t >= args.seconds:
                break

            chunk = read_available(fd, timeout_s=0.05, max_bytes=4096)
            if not chunk:
                continue

            host_time = time.time()
            bytes_seen += len(chunk)
            if raw_handle:
                raw_handle.write(chunk)
            if args.mode in ("hex", "both"):
                print_hex_chunk(chunk, args.hex_bytes)

            buffer.extend(chunk)
            packets, dropped = extract_packets(buffer)
            dropped_total += dropped
            for packet, decoded in packets:
                packet_index += 1
                if not decoded["checksum_ok"]:
                    bad_checksum_total += 1
                if all(value == 0 for value in decoded["counts"]):
                    all_zero_total += 1

                if csv_writer:
                    csv_writer.writerow([
                        host_time,
                        packet_index,
                        decoded["board_ms"],
                        decoded["checksum_ok"],
                        decoded["end_ok"],
                        decoded["status_hex"],
                        *decoded["counts"],
                        *[f"{v:.9f}" for v in decoded["microvolts"]],
                    ])

                if args.mode in ("packets", "both") and packet_index % max(1, args.packet_print_every) == 0:
                    print(format_packet_line(packet_index, decoded))

            if now - last_stats_t >= args.stats_every_s:
                dt = now - last_stats_t
                bps = (bytes_seen - bytes_last) / dt
                pps = (packet_index - packets_last) / dt
                print(
                    f"[stats] bytes/s={bps:8.1f} packets/s={pps:6.1f} "
                    f"packets={packet_index} dropped={dropped_total} "
                    f"bad_checksum={bad_checksum_total} all_zero_packets={all_zero_total}"
                )
                bytes_last = bytes_seen
                packets_last = packet_index
                last_stats_t = now

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if fd is not None:
            os.close(fd)
        if raw_handle:
            raw_handle.close()
        if csv_handle:
            csv_handle.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
