import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from pathlib import Path

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
SCRAPED_DIR = DATASET_DIR / 'scraped'
SCRAPED_DIR.mkdir(parents=True, exist_ok=True)

def stop(time_sleep):
    time.sleep(np.random.rand() + time_sleep)

def scraping_recipes(urls):
    driver = webdriver.Chrome()
    url = 'https://www.homebrewersassociation.org/beer-recipes/'
    driver.get(url)
    driver.maximize_window()

    account_button = driver.find_element(By.XPATH, '/html/body/div[1]/header/div[1]/div[3]/div/div/button')
    account_button.click()
    stop(0.5)
    login_button = driver.find_element(By.XPATH, '/html/body/div[1]/header/div[1]/div[3]/div/div/div/div[1]/a[1]')
    login_button.click()
    stop(5)

    email_fill = driver.find_element(By.XPATH, '/html/body/div[3]/div[2]/div/div[2]/div/div[2]/div/div[1]/div/input')
    email_fill.send_keys("patcharapol.yasamut@gmail.com")
    password = driver.find_element(By.XPATH, '/html/body/div[3]/div[2]/div/div[2]/div/div[2]/div/div[2]/div/input')
    password.send_keys('KeepGoing_2075')
    login_button2 = driver.find_element(By.XPATH, '/html/body/div[3]/div[2]/div/div[2]/div/div[2]/div/div[3]/button')
    login_button2.click()

    stop(20)

    filename = input('put filname: ')
    df_all = pd.DataFrame([], columns=['url', 'src', 'description', 'ingredients', 'directions', 'addition'])
    output_path = SCRAPED_DIR / f'{filename}.csv'
    df_all.to_csv(output_path, index=False)
    
    for i, url in enumerate(urls):
        print(i, url)
        all_text = {}
        all_text['url'] = url
        
        driver.get(url)
        stop(1)

        # Get Medal Title
        try:
            img = driver.find_element(By.CLASS_NAME, 'img-flag')
            all_text['src'] = img.get_attribute('src')
        except Exception as e:
            all_text['src'] = 'Not Found'
        
        # Get Recipe Description => 2 options for 2 layout web
        try: 
            descrip_div = driver.find_element(By.CLASS_NAME, 'recipe-description')
            all_text['description'] = [descrip_div.text]
            if descrip_div.text == '':
                descrip_div = driver.find_element(By.CLASS_NAME, 'alternate-block')
                all_text['description'] = [descrip_div.text]
        except Exception as e:
            pass
        
        # Get Ingredients
        try:
            ingre_div = driver.find_element(By.CLASS_NAME, 'ingredients')
            all_text['ingredients'] = [ingre_div.text]
        except Exception as e:
            pass
        
        # Get Directions
        try:
            direct_div = driver.find_element(By.CLASS_NAME, 'directions')
            all_text['directions'] = [direct_div.text]
        except Exception as e:
            pass
        
        # Get Additional Recipe
        try:
            addition_div = driver.find_element(By.CLASS_NAME, 'recipe-additional')
            all_text['addition'] = [addition_div.text]
        except Exception as e:
            pass
        
        df_all = pd.read_csv(output_path)
        df = pd.DataFrame().from_dict(all_text)
        df_all = pd.concat([df_all, df])
        df_all.to_csv(output_path, index=False)
    return df_all


if __name__ == '__main__':
    all_urls_path = DATASET_DIR / 'scraped' / 'all_urls.csv'
    df_url = pd.read_csv(all_urls_path)
    urls = df_url['url'].unique()
    df_all_recipe = scraping_recipes(urls[:10])
    # df_all_recipe.to_csv('all_recipes.csv', index=False)