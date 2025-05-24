import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
import json

# Import functions from main.py
from main import scrape_steam_game, get_popular_steam_games, main

class TestSteamScraper(unittest.TestCase):
    
    def setUp(self):
        # Create sample HTML content for testing
        with open('sample_game_page.html', 'w', encoding='utf-8') as f:
            f.write("""
            <html>
                <div class="apphub_AppName">Test Game</div>
                <div class="game_description_snippet">This is a test game description</div>
                <div class="release_date"><div class="date">Jan 1, 2023</div></div>
                <div class="dev_row"><div class="summary"><a>Test Developer</a></div></div>
                <div class="app_tag">Action</div>
                <div class="app_tag">Adventure</div>
                <div class="app_tag">RPG</div>
                <div class="game_header_image_full" src="test_image.jpg"></div>
                <div class="game_purchase_price">$19.99</div>
                <div class="game_review_summary">Very Positive</div>
            </html>
            """)
            
        with open('sample_search_page.html', 'w', encoding='utf-8') as f:
            f.write("""
            <html>
                <a class="search_result_row" data-ds-appid="123456">Game 1</a>
                <a class="search_result_row" data-ds-appid="234567">Game 2</a>
                <a class="search_result_row" data-ds-appid="345678">Game 3</a>
                <a class="search_result_row" data-ds-appid="456789,567890">Game 4</a>
            </html>
            """)
            
    def tearDown(self):
        # Clean up sample files
        if os.path.exists('sample_game_page.html'):
            os.remove('sample_game_page.html')
        if os.path.exists('sample_search_page.html'):
            os.remove('sample_search_page.html')
    
    @patch('main.requests.get')
    @patch('main.time.sleep')  # Mock sleep to speed up tests
    def test_scrape_steam_game(self, mock_sleep, mock_requests_get):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = open('sample_game_page.html', 'r', encoding='utf-8').read()
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        # Call the function
        result = scrape_steam_game('123456')
        
        # Assertions
        self.assertEqual(result['AppID'], '123456')
        self.assertEqual(result['Name'], 'Test Game')
        self.assertEqual(result['About'], 'This is a test game description')
        self.assertEqual(result['ReleaseDate'], 'Jan 1, 2023')
        self.assertEqual(result['Developer'], 'Test Developer')
        self.assertTrue('Action' in result['Tags'])
        self.assertTrue('Adventure' in result['Tags'])
        self.assertTrue('RPG' in result['Tags'])
        self.assertEqual(result['Price'], '$19.99')
        self.assertEqual(result['Reviews'], 'Very Positive')
        
        # Verify requests.get was called with correct URL and headers
        mock_requests_get.assert_called_once()
        args, kwargs = mock_requests_get.call_args
        self.assertEqual(args[0], 'https://store.steampowered.com/app/123456')
        self.assertTrue('headers' in kwargs)
        
    @patch('main.requests.get')
    @patch('main.time.sleep')  # Mock sleep to speed up tests
    def test_get_popular_steam_games(self, mock_sleep, mock_requests_get):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = open('sample_search_page.html', 'r', encoding='utf-8').read()
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response
        
        # Call the function with just 1 page to test
        result = get_popular_steam_games(num_pages=1)
        
        # Assertions
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], '123456')
        self.assertEqual(result[1], '234567')
        self.assertEqual(result[2], '345678')
        self.assertEqual(result[3], '456789')  # Should take the first ID when multiple exist
        
        # Verify requests.get was called with correct URL and headers
        mock_requests_get.assert_called_once()
        args, kwargs = mock_requests_get.call_args
        self.assertEqual(args[0], 'https://store.steampowered.com/search/?filter=topsellers&page=1')
        
    @patch('main.scrape_steam_game')
    @patch('main.get_popular_steam_games')
    def test_main_function(self, mock_get_popular_steam_games, mock_scrape_steam_game):
        # Setup mock returns
        mock_get_popular_steam_games.return_value = ['123', '456', '789']
        
        mock_scrape_steam_game.side_effect = [
            {'AppID': '123', 'Name': 'Game 1', 'About': 'Description 1'},
            {'AppID': '456', 'Name': 'Game 2', 'About': 'Description 2'},
            {'AppID': '789', 'Name': 'Game 3', 'About': 'Description 3'}
        ]
        
        # Temporarily redirect output files
        original_csv_path = os.path.join('data', 'steam_games.csv')
        original_pickle_path = os.path.join('data', 'game_data.pkl')
        test_csv_path = os.path.join('data', 'test_steam_games.csv')
        test_pickle_path = os.path.join('data', 'test_game_data.pkl')
        
        # Create a patched version of main for testing
        @patch('main.os.path.join')
        def patched_main(mock_join):
            mock_join.side_effect = lambda dir, file: test_csv_path if file == 'steam_games.csv' else test_pickle_path
            return main()
            
        # Call the patched main function
        df = patched_main()
        
        # Assertions
        self.assertEqual(len(df), 3)
        self.assertEqual(df['AppID'].tolist(), ['123', '456', '789'])
        self.assertEqual(df['Name'].tolist(), ['Game 1', 'Game 2', 'Game 3'])
        
        # Ensure files were created
        self.assertTrue(os.path.exists(test_csv_path) or os.path.exists(original_csv_path))
        self.assertTrue(os.path.exists(test_pickle_path) or os.path.exists(original_pickle_path))
        
        # Cleanup test files
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)
        if os.path.exists(test_pickle_path):
            os.remove(test_pickle_path)

    @patch('main.requests.get')
    def test_error_handling_in_scrape_game(self, mock_requests_get):
        # Setup mock to raise an exception
        mock_requests_get.side_effect = Exception("Test error")
        
        # Call the function
        result = scrape_steam_game('123456')
        
        # Assertions
        self.assertEqual(result['AppID'], '123456')
        self.assertTrue('Error' in result['Name'])
        self.assertEqual(result['About'], "")
        
    @patch('main.requests.get')
    def test_error_handling_in_get_popular_games(self, mock_requests_get):
        # Setup mock to raise an exception
        mock_requests_get.side_effect = Exception("Test error")
        
        # Call the function
        result = get_popular_steam_games(num_pages=1)
        
        # Assertions
        self.assertEqual(result, [])  # Should return empty list on error

if __name__ == '__main__':
    unittest.main()