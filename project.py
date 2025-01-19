from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Initialize WebDriver
driver = webdriver.Chrome()

query = 'laptop'
file = 0
# Iterate through pages
for i in range(1, 50):
    driver.get(f"https://www.flipkart.com/search?q={query}&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&page={i}")
    elems = driver.find_elements(By.CLASS_NAME, "_75nlfW")
    print(f'{len(elems)} items found on page {i}')
    
    for elem in elems:
        d = elem.get_attribute('outerHTML')
        with open(f"laptops/{query}_{file}.html","w",encoding='utf-8') as f:
            f.write(d)
            file +=1
    
    time.sleep(6)

# Close the browser after completing all iterations
driver.quit()
