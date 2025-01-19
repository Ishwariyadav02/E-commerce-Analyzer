from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Initialize WebDriver
driver = webdriver.Chrome()

query = 'laptop'

# Iterate through pages
for i in range(1, 3):
    driver.get(f"https://www.flipkart.com/search?q={query}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&page={i}")
    elems = driver.find_elements(By.CLASS_NAME, "_75nlfW")
    print(f'{len(elems)} items found on page {i}')
    
    for elem in elems:
        print(elem.text)
       
        print(elem.get_attribute('outerHTML'))
    
    time.sleep(6)

# Close the browser after completing all iterations
driver.quit()
