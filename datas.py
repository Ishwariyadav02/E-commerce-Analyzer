from bs4 import BeautifulSoup
import os
import pandas as pd
import re

def clean_price(price_str):
    """Clean price string by removing currency symbol and converting to proper format"""
    if not price_str or price_str == 'No Price':
        return price_str
    # Remove the currency symbol and any non-numeric characters except commas
    cleaned = re.sub(r'[^\d,]', '', price_str)
    return cleaned

def clean_percentage(percentage_str):
    """Clean percentage string by removing % symbol and extra text"""
    if not percentage_str or 'No Discount' in percentage_str:
        return percentage_str
    # Extract just the number from strings like "45% off"
    match = re.search(r'(\d+)%', percentage_str)
    if match:
        return match.group(1)
    return percentage_str

# List to store the extracted data
data = []

# Directory containing HTML files
html_dir = "data"

# Iterate through files in the 'data' directory
for file in os.listdir(html_dir):
    with open(f"{html_dir}/{file}", encoding='utf-8') as f:
        html_doc = f.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    
    # Find all product containers
    products = soup.find_all('div', class_='_75nlfW')
    
    for product in products:
        # Initialize dictionary for current product
        product_specs = {
            'Product Name': 'No Product Name',
            'Processor': 'No Processor',
            'RAM': 'No RAM',
            'Storage': 'No Storage',
            'Operating System': 'No Operating System',
            'Display': 'No Display',
            'Warranty': 'No Warranty',
            'Price': 'No Price',
            'Original Price': 'No Original Price',
            'Discount': 'No Discount',
            'Rating': 'No Rating',
            'Number of Ratings': 'No Ratings Count',
            'Number of Reviews': 'No Reviews Count',
            'Image URL': 'No Image URL',
            'Product Link': 'No Product Link'
        }
        
        # Extract product name
        name_tag = product.find('div', class_='KzDlHZ')
        if name_tag:
            product_specs['Product Name'] = name_tag.text.strip()
        
        # Extract specifications from list
        specs_list = product.find('ul', class_='G4BRas')
        if specs_list:
            specs_items = specs_list.find_all('li', class_='J+igdf')
            for item in specs_items:
                text = item.text.strip()
                if 'Processor' in text:
                    product_specs['Processor'] = text
                elif 'RAM' in text:
                    product_specs['RAM'] = text
                elif 'SSD' in text or 'HDD' in text:
                    product_specs['Storage'] = text
                elif 'Operating System' in text:
                    product_specs['Operating System'] = text
                elif 'Display' in text:
                    product_specs['Display'] = text
                elif 'warranty' in text.lower():
                    product_specs['Warranty'] = text
        
        # Extract price
        price_tag = product.find('div', class_='Nx9bqj')
        if price_tag:
            product_specs['Price'] = clean_price(price_tag.text.strip())
            
        # Extract original price
        original_price_tag = product.find('div', class_='yRaY8j')
        if original_price_tag:
            product_specs['Original Price'] = clean_price(original_price_tag.text.strip())
            
        # Extract discount
        discount_tag = product.find('div', class_='UkUFwK')
        if discount_tag:
            product_specs['Discount'] = clean_percentage(discount_tag.text.strip())
        
        # Extract rating
        rating_tag = product.find('div', class_='XQDdHH')
        if rating_tag:
            # Remove any non-numeric characters except decimal point
            rating_text = rating_tag.text.strip()
            product_specs['Rating'] = re.sub(r'[^\d.]', '', rating_text)
            
        # Extract ratings and reviews count
        ratings_reviews = product.find('span', class_='Wphh3N')
        if ratings_reviews:
            text = ratings_reviews.text.strip()
            # Extract numbers using string manipulation
            if 'Ratings' in text:
                ratings_count = re.search(r'(\d+)\s*Ratings', text)
                if ratings_count:
                    product_specs['Number of Ratings'] = ratings_count.group(1)
            if 'Reviews' in text:
                reviews_count = re.search(r'(\d+)\s*Reviews', text)
                if reviews_count:
                    product_specs['Number of Reviews'] = reviews_count.group(1)
        
        # Extract image URL
        img_tag = product.find('img', class_='DByuf4')
        if img_tag and 'src' in img_tag.attrs:
            product_specs['Image URL'] = img_tag['src']
            
        # Extract product link
        link_tag = product.find('a', class_='CGtC98')
        if link_tag and 'href' in link_tag.attrs:
            product_specs['Product Link'] = 'https://www.flipkart.com' + link_tag['href']
        
        # Append the product data
        data.append(product_specs)

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_file = 'laptop_data.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')  # Using utf-8-sig encoding for better Excel compatibility

# Display the first few rows and basic information
print(f"\nScraped data saved to {output_file}")
print("\nFirst few rows of the data:")
print(df.head())
print("\nDataset information:")
print(df.info())