import time
import csv
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException

options = Options()
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)

try:
    driver.get("https://www.redfin.com/city/1826/MA/Boston")
    wait = WebDriverWait(driver, 10)
    
    links = []
    page = 1  
    
    while True:
        print(f"\nProcessing Page {page}...")
        try:
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".bp-Homecard")))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2) 
            
            cards = driver.find_elements(By.CSS_SELECTOR, ".bp-Homecard")
            print(f"Found {len(cards)} cards on this page.")
            
            for index, card in enumerate(cards, start=1):
                try:
                    link = card.find_element(By.CSS_SELECTOR, "a.link-and-anchor")
                    href = link.get_attribute("href")
                    links.append(href)
                    print(f"Collected Link {len(links)}: {href}")
                
                except NoSuchElementException:  print(f"Card {index}: Link element not found.")
                except Exception as e:          print(f"Card {index}: Unexpected error: {e}")
            
            try:
                next_button = wait.until(EC.element_to_be_clickable((
                    By.CSS_SELECTOR, 
                    "button.bp-Button.PageArrow.clickable.Pagination__button.PageArrow__direction--next.bp-Button__type--ghost.bp-Button__size--compact.bp-Button__icon-only"
                )))
                next_button.click()
                page += 1 
                print("Navigated to the next page.")
                
                wait.until(EC.staleness_of(cards[0]))
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".bp-Homecard")))

                time.sleep(2)
            
            except Exception as e:
                m = {
                    TimeoutException: "Next button not found. Assuming last page reached.",
                    NoSuchElementException: "Next button does not exist. Assuming last page reached.",
                    ElementClickInterceptedException: "Next button could not be clicked. It might be obscured by another element.",
                }
                message = m.get(type(e), f"An unexpected error occurred while clicking 'Next': {e}")
                print(message)
                break
        
        except Exception as e:
            m = { TimeoutException: "Home cards did not load in time. Exiting loop." }
            message = m.get(type(e), f"An unexpected error occurred: {e}")
            print(message)
            break
    
    with open('links.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'URL'])
        for idx, link in enumerate(links, start=1):
            writer.writerow([idx, link])
    
    print(f"\nAll collected links have been saved. Total links: {len(links)}")

finally:
    driver.quit()
