import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def scrape_steam_game(app_id):
    """Scrape details for a single Steam game by app_id"""
    url = f"https://store.steampowered.com/app/{app_id}"
    
    # Add headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    # Add random delay to avoid rate limiting
    time.sleep(random.uniform(1, 3))
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract game title
        title = soup.find('div', class_='apphub_AppName')
        game_title = title.text.strip() if title else "Unknown Title"
        
        # Extract game description
        description_div = soup.find('div', class_='game_description_snippet')
        description = description_div.text.strip() if description_div else ""
        
        # Extract release date
        release_date_div = soup.select_one('.release_date .date')
        release_date = release_date_div.text.strip() if release_date_div else "Unknown Date"
        
        # Extract developer
        developer_div = soup.select_one('.dev_row .summary a')
        developer = developer_div.text.strip() if developer_div else "Unknown Developer"
        
        # Extract tags/genres
        tags_divs = soup.select('.app_tag')
        tags = [tag.text.strip() for tag in tags_divs if tag.text.strip()]
        
        # Extract header image
        header_img = soup.select_one('.game_header_image_full')
        image_url = header_img['src'] if header_img and 'src' in header_img.attrs else ""
        
        # Extract price
        price_div = soup.select_one('.game_purchase_price') or soup.select_one('.discount_final_price')
        price = price_div.text.strip() if price_div else "Unknown Price"
        
        # Extract reviews summary
        reviews_div = soup.select_one('.game_review_summary')
        reviews = reviews_div.text.strip() if reviews_div else "No Reviews"
        
        return {
            'AppID': app_id,
            'Name': game_title,
            'About': description,
            'ReleaseDate': release_date,
            'Developer': developer,
            'Tags': ', '.join(tags[:10]),  # Limit to first 10 tags
            'HeaderImage': image_url,
            'Price': price,
            'Reviews': reviews
        }
    
    except Exception as e:
        print(f"Error scraping game {app_id}: {e}")
        return {
            'AppID': app_id,
            'Name': f"Error: {str(e)}",
            'About': "",
            'ReleaseDate': "",
            'Developer': "",
            'Tags': "",
            'HeaderImage': "",
            'Price': "",
            'Reviews': ""
        }

def get_popular_steam_games(num_pages=5):
    """Get a list of popular Steam game IDs from the top sellers page"""
    app_ids = []
    
    for page in range(1, num_pages + 1):
        url = f"https://store.steampowered.com/search/?filter=topsellers&page={page}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all game elements
            game_elements = soup.select('a.search_result_row')
            
            for game in game_elements:
                if 'data-ds-appid' in game.attrs:
                    app_id = game['data-ds-appid']
                    # Some games have multiple app IDs separated by commas, take the first one
                    app_id = app_id.split(',')[0]
                    app_ids.append(app_id)
            
            print(f"Page {page}: Found {len(game_elements)} games")
            
            # Add a delay between pages
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
    
    return app_ids

def main():
    # Get list of popular game IDs
    print("Getting list of popular Steam games...")
    app_ids = get_popular_steam_games(num_pages=5)  # Adjust number of pages as needed
    
    print(f"Found {len(app_ids)} games. Starting detailed scraping...")
    
    # Scrape details for each game
    games_data = []
    total_games = len(app_ids)
    
    for i, app_id in enumerate(app_ids[:100]):  # Limit to 100 games
        print(f"Scraping game {i+1}/{min(total_games, 100)}: App ID {app_id}")
        game_data = scrape_steam_game(app_id)
        games_data.append(game_data)
    
    # Save data to CSV
    df = pd.DataFrame(games_data)
    csv_path = os.path.join('data', 'steam_games.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Also save as pickle for easier loading
    pickle_path = os.path.join('data', 'game_data.pkl')
    df.to_pickle(pickle_path)
    print(f"Data saved to {pickle_path}")
    
    return df

if __name__ == "__main__":
    main()