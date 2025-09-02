#!/bin/bash

SCRIPT="redfin-process.py"

locations() {
    relay_list=$(mullvad relay list)
    locations=($(echo "$relay_list" | grep -oP '^\w.+\s\(\K\w+(?=\))' | sort -u))
}

change_vpn_location() {
    locations
    current=$(mullvad status | grep -oP '(?<=Connected to ).*?(?= \()' | awk '{print $1}')

    while true; do
        RANDOM_LOCATION=${locations[$RANDOM % ${#locations[@]}]}
        if [ "$RANDOM_LOCATION" != "$current" ]; then
            break
        fi
    done

    echo "Changing VPN location to $RANDOM_LOCATION..."

    mullvad relay set location "$RANDOM_LOCATION"
    mullvad connect

    sleep 3

    mullvad status
}

while true; do
    python3 "$SCRIPT"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script completed successfully."
        exit 0
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "Script exited with code $EXIT_CODE due to 403 ERROR. Changing VPN location."
        change_vpn_location
        echo "Restarting script in 3 seconds..."
        sleep 3
    else
        echo "Script exited with code $EXIT_CODE. Restarting in 3 seconds..."
        sleep 3
    fi
done
