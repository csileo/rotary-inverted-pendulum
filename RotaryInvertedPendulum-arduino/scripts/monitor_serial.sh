#!/bin/bash
# Simple serial monitor script
# Usage: ./monitor_serial.sh <port> <baud_rate> [duration_seconds]

PORT=${1:-/dev/cu.usbserial-10}
BAUD=${2:-115200}
DURATION=${3:-10}

echo "Monitoring $PORT at $BAUD baud..."
echo "Waiting for Arduino to reset..."
echo "---"

# Open port once with file descriptor to avoid double reset
exec 3<>"$PORT"
stty -f "$PORT" "$BAUD" cs8 -cstopb -parenb raw -echo

# Flush any old buffered data and wait for Arduino reset
timeout 0.1 cat <&3 >/dev/null 2>&1 || true
sleep 2

# Read from the already-open file descriptor
timeout "$DURATION" cat <&3 2>/dev/null || true

# Close file descriptor
exec 3<&-

echo ""
echo "---"
echo "Monitoring stopped."
