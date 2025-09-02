all: model

venv:
	python3 -m venv venv

install: venv requirements.txt
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Scrape links from Redfin and save to CSV
scrape: install redfin-scrape.py
	. venv/bin/activate && python redfin-scrape.py

# Check if Mullvad CLI is installed
check_mullvad:
	@command -v mullvad >/dev/null 2>&1 || { echo >&2 "Error: Mullvad CLI Required"; exit 1; }

# Extract and process listings from scraped links using the runner script
extract: install check_mullvad runner.sh
	./runner.sh

# Train and evaluate the housing price prediction model
model: install 506-final-model.py
	. venv/bin/activate && python 506-final-model.py

clean:
	rm -rf venv

.PHONY: all install scrape extract model clean venv
