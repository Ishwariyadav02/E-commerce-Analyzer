from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Set up the WebDriver (Chrome in this case)
driver = webdriver.Chrome()

try:
    # Navigate to a website
    driver.get("https://www.google.com")

    # Locate the search box
    search_box = driver.find_element(By.NAME, "q")

    # Enter text and perform a search
    search_box.send_keys("Selenium tutorial" + Keys.RETURN)

    # Wait for results and capture them
    results = driver.find_elements(By.CSS_SELECTOR, "h3")

    # Print the titles of the results
    for result in results:
        print(result.text)
finally:
    # Close the browser
    driver.quit()
