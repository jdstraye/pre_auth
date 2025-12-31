#!/bin/bash

# Prevent sleep/suspend and disable screensaver
systemd-inhibit --what=idle:sleep --who="Focus Script" --why="Keep computer awake" \
    --mode=block xset s off -dpms

# Bring VS Code to front every 5 minutes
while true; do
    wmctrl -a "code" || wmctrl -a "Visual Studio Code"
    sleep 300
done