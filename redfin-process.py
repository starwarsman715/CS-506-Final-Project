import os
import sys
import time
import csv
import pandas as pd
import logging
import re
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver import Remote, FirefoxOptions
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException


INPUT_CSV = 'redfin_boston_links.csv' 
OUTPUT_CSV = 'redfin_processed_listings.csv'
HEADLESS = True        
LOG_FILE = 'process.log' 
MAX_RETRIES = 3     
SLEEP_BETWEEN_LISTINGS = 2 

AUTH = 'brd-customer-hl_261b1f7d-zone-506:conwsl48rl99'
SBR_WEBDRIVER = f'https://{AUTH}@zproxy.lum-superproxy.io:9515'

logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def init_driver(headless=True):
    options = FirefoxOptions()
    if headless: options.add_argument("--headless")
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
    return driver

def init_remote_driver(headless=True):
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--headless")
    
    try:
        logging.info(f"Establishing remote WebDriver connection to {SBR_WEBDRIVER}")
        driver = Remote(command_executor=SBR_WEBDRIVER, options=chrome_options)
        logging.info("Remote WebDriver initialized successfully.")
        return driver
    except Exception as e:
        logging.critical(f"Failed to initialize Remote WebDriver: {e}")
        raise

def load_input_csv(input_csv):
    if not os.path.exists(input_csv):
        logging.error(f"Input CSV '{input_csv}' not found.")
        raise FileNotFoundError(f"Input CSV '{input_csv}' not found.")
    df = pd.read_csv(input_csv)
    if 'Index' not in df.columns or 'URL' not in df.columns:
        logging.error("Input CSV must contain 'Index' and 'URL' columns.")
        raise ValueError("Input CSV must contain 'Index' and 'URL' columns.")
    return df

def load_output_csv(output_csv):
    if os.path.exists(output_csv):
        processed_df = pd.read_csv(output_csv)
        processed_listings = set(processed_df['listing_number'].astype(str))
        logging.info(f"Loaded {len(processed_listings)} processed listings from '{output_csv}'.")
        return processed_df, processed_listings
    else:
        headers = ['listing_number', 'price', 'beds', 'baths', 'sq_ft', 'description', 'key_details', 'walkscore']
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        logging.info(f"Initialized new output CSV '{output_csv}'.")
        return pd.DataFrame(columns=headers), set()

def extract_listing_number(url):
    match = re.search(r'/home/(\d+)', url)
    if match: return match.group(1)
    else:
        logging.warning(f"Could not extract listing number from URL: {url}")
        return None

def clean_data(data):
    if data.get('price'):   data['price'] = re.sub(r'[^\d]', '', data['price'])
    if data.get('beds'):    data['beds'] = re.sub(r'[^\d]', '', data['beds']) 
    if data.get('baths'):   data['baths'] = re.sub(r'[^\d.]', '', data['baths'])
    if data.get('sq_ft'):   data['sq_ft'] = re.sub(r'[^\d]', '', data['sq_ft'])
    return data

def scroll_to_element(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({ behavior: 'smooth', block: 'center' });", element)
    time.sleep(1) 
    
def extract_walkscore(driver, wait, listing_number):
    try:
        mm = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "MiniMapSection")))
        logging.info(f"Located MiniMapSection for Listing {listing_number}.")

        scroll_to_element(driver, mm)
        logging.info(f"Scrolled to MiniMapSection for Listing {listing_number}.")

        walkscore_pills = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".walkscore-pills")))
        logging.info(f"Located walkscore-pills for Listing {listing_number}.")

        walkscore_elems = walkscore_pills.find_elements(By.CSS_SELECTOR, "span.pill.walkscore")
        logging.info(f"Found {len(walkscore_elems)} walkscore elements for Listing {listing_number}.")

        walkscores = []
        for elem in walkscore_elems:
            try:
                svg_elem = elem.find_element(By.CSS_SELECTOR, "svg.bp-SvgIcon")
                svg_classes = svg_elem.get_attribute('class').split()
                if len(svg_classes) >= 2: category = svg_classes[1].capitalize()
                else: category = 'Unknown'

                text = elem.text.strip()
                walkscores.append(f"{category}: {text}")
            except NoSuchElementException:
                logging.warning(f"SVG element not found within walkscore pill for Listing {listing_number}.")
                continue

        walkscore = ' | '.join(walkscores) if walkscores else None
        logging.info(f"Extracted walkscore for Listing {listing_number}: {walkscore}")
        return walkscore

    except TimeoutException:
        logging.error(f"Timeout while trying to locate walkscore elements for Listing {listing_number}.")
        return None
    except Exception as e:
        logging.error(f"Error extracting walkscore for Listing {listing_number}: {e}")
        return None

def extract_listing_data(driver, wait, listing_number):
    data = {}

    try:
        price_elem = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.stat-block[data-rf-test-id='abp-price'] div.statsValue")
        ))
        data['price'] = price_elem.text.strip()
        logging.info(f"Extracted price for Listing {listing_number}: {data['price']}")
    except (NoSuchElementException, TimeoutException):
        data['price'] = None
        logging.warning(f"Price not found for Listing {listing_number}.")

    try:
        beds_elem = driver.find_element(By.CSS_SELECTOR, "div.stat-block[data-rf-test-id='abp-beds'] div.statsValue")
        data['beds'] = beds_elem.text.strip()
        logging.info(f"Extracted beds for Listing {listing_number}: {data['beds']}")
    except NoSuchElementException:
        data['beds'] = None
        logging.warning(f"Beds not found for Listing {listing_number}.")

    try:
        baths_elem = driver.find_element(By.CSS_SELECTOR, "div.stat-block[data-rf-test-id='abp-baths'] div.statsValue")
        data['baths'] = baths_elem.text.strip()
        logging.info(f"Extracted baths for Listing {listing_number}: {data['baths']}")
    except NoSuchElementException:
        data['baths'] = None
        logging.warning(f"Baths not found for Listing {listing_number}.")

    try:
        sqft_elem = driver.find_element(By.CSS_SELECTOR, "div.stat-block[data-rf-test-id='abp-sqFt'] span.statsValue")
        data['sq_ft'] = sqft_elem.text.strip()
        logging.info(f"Extracted square footage for Listing {listing_number}: {data['sq_ft']}")
    except NoSuchElementException:
        data['sq_ft'] = None
        logging.warning(f"Square footage not found for Listing {listing_number}.")

    try:
        desc_elem = driver.find_element(By.ID, "marketing-remarks-scroll")
        data['description'] = desc_elem.text.strip()
        logging.info(f"Extracted description for Listing {listing_number}.")
    except NoSuchElementException:
        data['description'] = None
        logging.warning(f"Description not found for Listing {listing_number}.")

    try:
        key_details_elems = driver.find_elements(By.CSS_SELECTOR, "div.keyDetailsList div.keyDetails-row")
        key_details = {}
        for elem in key_details_elems:
            try:
                svg_elem = elem.find_element(By.CSS_SELECTOR, "span.keyDetails-label svg")
                svg_classes = svg_elem.get_attribute('class').split()
                if len(svg_classes) >= 2: key = svg_classes[1].replace('-', ' ').capitalize() 
                else: key = 'Unknown'

                value_elem = elem.find_element(By.CSS_SELECTOR, "div.keyDetails-value")
                value = value_elem.text.strip()
                key_details[key] = value
            except NoSuchElementException:
                logging.warning(f"Key detail not found in Listing {listing_number}.")
                continue

        data['key_details'] = '; '.join([f"{k}: {v}" for k, v in key_details.items()]) if key_details else None
        logging.info(f"Extracted key_details for Listing {listing_number}: {data['key_details']}")
    except NoSuchElementException:
        data['key_details'] = None
        logging.warning(f"Key details not found for Listing {listing_number}.")

    walkscore = extract_walkscore(driver, wait, listing_number)
    data['walkscore'] = walkscore

    return data

def check_for_403_error(driver):
    time.sleep(3)
    try:
        h1_elem = driver.find_element(By.TAG_NAME, 'h1')
        if '403 ERROR' in h1_elem.text.strip():
            h2_elem = driver.find_element(By.TAG_NAME, 'h2')
            if 'The request could not be satisfied.' in h2_elem.text.strip():
                return True
    except NoSuchElementException:
        print("NO SUCH ELEMENT")
        pass

    if 'Generated by cloudfront' in driver.page_source:
        return True

    return False

def process_listings():
    try:
        input_df = load_input_csv(INPUT_CSV)
    except Exception as e:
        logging.critical(f"Failed to load input CSV: {e}")
        return

    try:
        output_df, processed_listings = load_output_csv(OUTPUT_CSV)
    except Exception as e:
        logging.critical(f"Failed to load or initialize output CSV: {e}")
        return

    total_listings = len(input_df)
    logging.info(f"Total listings to process: {total_listings}")
    processed_count = 0

    for index, row in input_df.iterrows():
        index_number = row['Index']
        url = row['URL']
        listing_number = extract_listing_number(url)

        if not listing_number:
            logging.warning(f"Skipping Index {index_number} due to missing listing number.")
            continue

        if listing_number in processed_listings:
            logging.info(f"Listing {listing_number} already processed. Skipping.")
            continue

        logging.info(f"Processing Listing {listing_number}: {url}")

        retries = 0
        while retries < MAX_RETRIES:
            try:
                driver = init_driver(headless=HEADLESS)
                wait = WebDriverWait(driver, 20)
                driver.get(url)
                logging.info(f"Navigated to Listing {listing_number} URL.")

                if check_for_403_error(driver):
                    print('403 ERROR')
                    logging.error(f"403 ERROR encountered for Listing {listing_number}. Exiting script.")
                    driver.quit()
                    sys.exit(1)
                    
                try:
                    close_button = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CLASS_NAME, 'bp-CloseButton')))
                    close_button.click()
                    logging.info("Closed modal using 'button.bp-CloseButton'")
                except:
                    pass

                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.stat-block[data-rf-test-id='abp-price']")))
                logging.info(f"Main content loaded for Listing {listing_number}.")

                data = extract_listing_data(driver, wait, listing_number)
                data = clean_data(data)
                data['listing_number'] = listing_number

                ordered_data = {
                    'listing_number': data.get('listing_number'),
                    'price': data.get('price'),
                    'beds': data.get('beds'),
                    'baths': data.get('baths'),
                    'sq_ft': data.get('sq_ft'),
                    'description': data.get('description'),
                    'key_details': data.get('key_details'),
                    'walkscore': data.get('walkscore'),
                }

                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=ordered_data.keys())
                    writer.writerow(ordered_data)

                logging.info(f"Successfully processed Listing {listing_number}.")
                processed_count += 1

                driver.quit()
                break  

            except (TimeoutException, NoSuchElementException, ElementClickInterceptedException) as e:
                retries += 1
                logging.warning(f"Error processing Listing {listing_number} (Attempt {retries}): {e}")
                try:
                    driver.quit()
                except:
                    pass 
                time.sleep(5) 
            except Exception as e:
                retries += 1
                logging.error(f"Unexpected error processing Listing {listing_number} (Attempt {retries}): {e}")
                time.sleep(5) 

        if retries == MAX_RETRIES:
            logging.error(f"Failed to process Listing {listing_number} after {MAX_RETRIES} attempts.")
            continue 

        time.sleep(SLEEP_BETWEEN_LISTINGS)

    logging.info(f"Processing completed. Total listings processed: {processed_count}")
    sys.exit(0) 

if __name__ == "__main__":
    process_listings()
