import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from pathlib import Path

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
SCRAPED_DIR = DATASET_DIR / 'scraped'
SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

def stop(time_sleep):
    time.sleep(np.random.rand() + time_sleep)

def get_recipe_url_in_page(driver, url, page_number):
    try:
        driver.get(url)
        # Wait until all articles are loaded (max 15 seconds)
        wait = WebDriverWait(driver, 60)
        wait.until(EC.presence_of_all_elements_located((By.XPATH, '//article')))
        all_articles = driver.find_elements(By.XPATH, '//article')
        all_recipe = []
        for article_element in all_articles:
            style_path = './div/a/span/span'
            final_results_path = './div/div'
            try:
                style_elem = article_element.find_element(By.XPATH, style_path)
                style_text = style_elem.text
                # Get the anchor tag that contains both the name and the href
                anchor_elem = article_element.find_element(By.XPATH, './div/header/p/a')
                name_text = anchor_elem.text
                url_recipe = anchor_elem.get_attribute("href")
                final_result_text = [t.text for t in article_element.find_elements(By.XPATH, final_results_path)]
                all_recipe.append([page_number, style_text, name_text, final_result_text, url_recipe])
            except Exception as e:
                print(f"Error extracting recipe data: {e}")
                continue
        return all_recipe
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def process_pages_with_driver(page_urls_with_numbers):
    """Process a batch of pages with a single Chrome instance."""
    driver = webdriver.Chrome()
    driver.maximize_window()
    all_recipe = []
    try:
        for page_num, url in page_urls_with_numbers:
            result = get_recipe_url_in_page(driver, url, page_num)
            all_recipe.extend(result)
            print(f"Completed: Page {page_num} - {url}")
    finally:
        driver.quit()  # Close browser after processing all pages
    return all_recipe

if __name__ == '__main__':

    start_page, end_page = 2, 84
    first_page_url = 'https://homebrewersassociation.org/homebrew-recipes/?aha-recipes%5Bmenu%5D%5Bbev_type%5D=Beer&aha-recipes%5Brange%5D%5Babv_num%5D=%3A20'
    other_page_url = [f'https://homebrewersassociation.org/homebrew-recipes/?aha-recipes%5Bpage%5D={int(num)}&aha-recipes%5Bmenu%5D%5Bbev_type%5D=Beer&aha-recipes%5Brange%5D%5Babv_num%5D=%3A20' for num in range(start_page, end_page+1)]
    all_page_url = [first_page_url] + other_page_url
    # all_page_url = all_page_url[:16]

    # Create list of (page_number, url) tuples - page 1 is first_page_url, then pages 2, 3, 4...
    page_numbers = [1] + list(range(start_page, start_page + len(all_page_url) - 1))
    all_page_url_with_numbers = list(zip(page_numbers, all_page_url))

    # Distribute pages equally across 4 threads
    num_pages = len(all_page_url_with_numbers)
    num_threads = 12
    pages_per_thread = (num_pages + num_threads - 1) // num_threads  # ceiling division

    # Split pages into batches for each thread
    page_batches = [all_page_url_with_numbers[i*pages_per_thread:(i+1)*pages_per_thread] for i in range(num_threads)]

    print(f"Processing {num_pages} pages with {num_threads} threads")
    print(f"Pages per thread: {pages_per_thread}")
    for i, batch in enumerate(page_batches):
        print(f"  Thread {i}: {len(batch)} pages")

    # Process page batches using ThreadPoolExecutor
    all_url = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_pages_with_driver, batch) for batch in page_batches]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                print(f"[Thread-{i}] Completed scraping {len(result)} recipes")
                all_url.extend(result)
            except Exception as e:
                print(f"[Thread-{i}] Error: {e}")

    df_all = pd.DataFrame(all_url, columns=['page_number', 'style', 'beer_name', 'final_results', 'url'])
    # Save only the URL column to CSV
    output_file = SCRAPED_DIR / 'all_urls.csv'
    df_all[['url']].to_csv(output_file, index=False)
    print(f"Saved {len(df_all)} recipes to {output_file}")