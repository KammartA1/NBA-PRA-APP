#!/bin/bash
# NBA PrizePicks Scraper — systemd setup
# Run: sudo bash services/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$SCRIPT_DIR/nba-prizepicks-scraper.service"

echo "=== NBA PrizePicks Scraper Setup ==="
echo "Project: $PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv "$PROJECT_DIR/venv" 2>/dev/null || python3 -m venv "$PROJECT_DIR/venv"
fi

# Install dependencies
echo "Installing dependencies..."
"$PROJECT_DIR/venv/bin/pip" install -q --upgrade pip
"$PROJECT_DIR/venv/bin/pip" install -q requests sqlalchemy curl_cffi

# Create directories
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data"

# Install systemd service
echo "Installing systemd service..."
cp "$SERVICE_FILE" /etc/systemd/system/nba-prizepicks-scraper.service
systemctl daemon-reload
systemctl enable nba-prizepicks-scraper
systemctl start nba-prizepicks-scraper

echo ""
echo "=== Setup complete ==="
echo "Service status:"
systemctl status nba-prizepicks-scraper --no-pager -l || true
echo ""
echo "Useful commands:"
echo "  systemctl status nba-prizepicks-scraper"
echo "  journalctl -u nba-prizepicks-scraper -f"
echo "  tail -f $PROJECT_DIR/logs/prizepicks_scraper.log"
