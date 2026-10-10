# Convert to .py to run

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Global HTTP headers to mimic a browser request
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def fetch_page_content(url, headers=HEADERS):
    """
    Fetches the content of a given URL with robust error handling.

    Args:
        url (str): The URL to fetch.
        headers (dict): HTTP headers to use for the request.

    Returns:
        BeautifulSoup object or None: Parsed HTML content if successful, None otherwise.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        print(f"Successfully fetched: {url}")
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.HTTPError as errh:
        print (f"Http Error for {url}: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print (f"Error Connecting to {url}: {errc}")
    except requests.exceptions.Timeout as errt:
        print (f"Timeout Error for {url}: {errt}")
    except requests.exceptions.RequestException as err:
        print (f"Oops: Something Else for {url}: {err}")
    return None

def extract_itsjustabadday_article_data(article_soup):
    """
    Extracts article details from a single 'article' BeautifulSoup object for itsjustabadday.com.

    Args:
        article_soup (BeautifulSoup tag): The BeautifulSoup tag object representing an <article>.

    Returns:
        dict: A dictionary containing extracted article information.
    """
    article_info = {
        'article_link': 'N/A',
        'article_title': 'N/A',
        'author': 'N/A',
        'categories': 'N/A',
        'date': 'N/A',
        'comments_tag': 'N/A'
    }

    # Extract Article Title and Link from h3.entry-title
    entry_title_h3 = article_soup.find('h3', class_='entry-title')
    if entry_title_h3:
        title_link_a = entry_title_h3.find('a', href=True)
        if title_link_a:
            article_info['article_title'] = title_link_a.get_text(strip=True)
            article_info['article_link'] = title_link_a['href']

    # Extract details from entry-meta div
    entry_meta_div = article_soup.find('div', class_='entry-meta')
    if entry_meta_div:
        # Author
        author_span = entry_meta_div.find('span', class_='entry-author')
        if author_span and author_span.find('a'):
            article_info['author'] = author_span.find('a').get_text(strip=True)

        # Categories
        categories_span = entry_meta_div.find('span', class_='entry-categories cat-links')
        if categories_span:
            category_links = categories_span.find_all('a', href=True)
            if category_links:
                article_info['categories'] = ', '.join([link.get_text(strip=True) for link in category_links])

        # Date
        date_span = entry_meta_div.find('span', class_='entry-date')
        if date_span:
            date_a_tag = date_span.find('a')
            if date_a_tag:
                article_info['date'] = date_a_tag.get_text(strip=True)
            else:
                article_info['date'] = date_span.get_text(strip=True)

        # Comments Tag
        comments_span = entry_meta_div.find('span', class_='entry-comments')
        if comments_span and comments_span.find('a'):
            article_info['comments_tag'] = comments_span.find('a').get_text(strip=True)
        elif comments_span:
            article_info['comments_tag'] = comments_span.get_text(strip=True)
    return article_info

def extract_psohoward_article_data(article_soup):
    """
    Extracts article details from a single 'article' BeautifulSoup object for psohoward.com.

    Args:
        article_soup (BeautifulSoup tag): The BeautifulSoup tag object representing an <article>.

    Returns:
        dict: A dictionary containing extracted article information including peripheral details,
              likes count, and article content.
    """
    article_info = {
        'article_link': 'N/A',
        'article_title': 'N/A',
        'author': 'N/A',
        'categories': 'N/A',
        'date': 'N/A',
        'comments_count': 'N/A',
        'likes_count': 0,
        'article_content': 'N/A'
    }

    # Extract Article Title and Link
    entry_title_h1 = article_soup.find('h1', class_='entry-title')
    if entry_title_h1:
        title_link_a = entry_title_h1.find('a', href=True)
        if title_link_a:
            article_info['article_title'] = title_link_a.get_text(strip=True)
            article_info['article_link'] = title_link_a['href']

    # Extract details from entry-meta footer
    entry_meta_footer = article_soup.find('footer', class_='entry-meta')
    if entry_meta_footer:
        # Author
        author_span = entry_meta_footer.find('span', class_='author vcard')
        if author_span and author_span.find('a'):
            article_info['author'] = author_span.find('a').get_text(strip=True)

        # Categories
        categories_span = entry_meta_footer.find('span', class_='tags-links')
        if categories_span:
            category_links = categories_span.find_all('a', href=True)
            if category_links:
                article_info['categories'] = ', '.join([link.get_text(strip=True) for link in category_links])

        # Date
        date_span = entry_meta_footer.find('span', class_='entry-date')
        if date_span:
            date_a_tag = date_span.find('a')
            if date_a_tag:
                article_info['date'] = date_a_tag.get_text(strip=True)
            else:
                article_info['date'] = date_span.get_text(strip=True)

        # Comments Count
        comments_a = entry_meta_footer.find('a', href=lambda href: href and '#comments' in href)
        if comments_a:
            comments_text = comments_a.get_text(strip=True)
            if 'Comment' in comments_text:
                try:
                    article_info['comments_count'] = int(comments_text.split(' ')[0])
                except ValueError:
                    article_info['comments_count'] = 0
            elif 'No Comments' in comments_text:
                article_info['comments_count'] = 0

    # Likes Count
    likes_text_span = article_soup.find('span', class_='wpl-count-text')
    if likes_text_span and likes_text_span.get_text(strip=True) == 'Be the first to like this.':
        article_info['likes_count'] = 0
    else:
        likes_number_span = article_soup.find('span', class_='wpl-count-number')
        if likes_number_span:
            try:
                article_info['likes_count'] = int(likes_number_span.get_text(strip=True))
            except ValueError:
                article_info['likes_count'] = 0

    # Extract Article Content
    content_div = article_soup.find('div', class_='entry-content clear')
    if content_div:
        paragraphs = content_div.find_all('p')
        if paragraphs:
            article_info['article_content'] = ' '.join(filter(None, [p.get_text(strip=True) for p in paragraphs]))
    return article_info

def scrape_blog(base_url, num_pages, article_extractor_func, start_page=1):
    """
    Scrapes articles from a multi-page blog.

    Args:
        base_url (str): The base URL of the blog (e.g., 'https://itsjustabadday.com/').
        num_pages (int): The total number of pages to scrape.
        article_extractor_func (function): The function to use for extracting article data
                                           from a single BeautifulSoup article tag.
        start_page (int): The starting page number for scraping (default is 1).

    Returns:
        list: A list of dictionaries, each containing article data.
    """
    all_articles_data = []
    for page_num in range(start_page, num_pages + 1):
        print(f"Processing page: {page_num}/{num_pages}")

        if page_num == 1:
            page_url = base_url
        else:
            page_url = f"{base_url}page/{page_num}/"
        print(f"  Fetching data from: {page_url}")

        soup = fetch_page_content(page_url)
        if soup:
            print("  HTML content parsed successfully.")
            all_article_elements = soup.find_all('article')
            print(f"  Found {len(all_article_elements)} article tags on page {page_num}.")

            for article_tag in all_article_elements:
                article_data = article_extractor_func(article_tag)
                all_articles_data.append(article_data)
        else:
            print(f"  Could not process page {page_num}. Skipping to next.")
            continue # Skip to the next page
    return all_articles_data

def scrape_article_full_content_itsjustabadday(article_links):
    """
    Scrapes the full content for a list of article links specific to itsjustabadday.com.

    Args:
        article_links (list): A list of unique article URLs.

    Returns:
        list: A list of dictionaries, each with 'article_link' and 'article_content'.
    """
    all_article_contents = []
    for i, link in enumerate(article_links):
        print(f"Processing article {i+1}/{len(article_links)}: {link}")

        try:
            response = requests.get(link, headers=HEADERS)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            print(f"  Successfully fetched content from: {link}")

            soup = BeautifulSoup(response.text, 'html.parser')
            # This class is specific to itsjustabadday.com's main content area
            article_content_div = soup.find('div', class_='entry-content')
            article_text = 'N/A'
            if article_content_div:
                article_text = article_content_div.get_text(separator=' ', strip=True)

            all_article_contents.append({
                'article_link': link,
                'article_content': article_text
            })
            print(f"  Extracted content for: {link}")

        except requests.exceptions.HTTPError as errh:
            print(f"  Http Error for {link}: {errh}")
            all_article_contents.append({
                'article_link': link,
                'article_content': f"HTTP Error: {errh}"
            })
        except requests.exceptions.ConnectionError as errc:
            print(f"  Error Connecting to {link}: {errc}")
            all_article_contents.append({
                'article_link': link,
                'article_content': f"Connection Error: {errc}"
            })
        except requests.exceptions.Timeout as errt:
            print(f"  Timeout Error for {link}: {errt}")
            all_article_contents.append({
                'article_link': link,
                'article_content': f"Timeout Error: {errt}"
            })
        except requests.exceptions.RequestException as err:
            print(f"  Oops: Something Else for {link}: {err}")
            all_article_contents.append({
                'article_link': link,
                'article_content': f"Request Exception: {err}"
            })
    return all_article_contents

if __name__ == "__main__":
    # --- Scrape itsjustabadday.com blog ---
    print("--- Starting scraping for itsjustabadday.com ---")
    itsjustabadday_base_url = 'https://itsjustabadday.com/'
    itsjustabadday_num_pages = 38

    # Scrape main article metadata for itsjustabadday.com
    itsjustabadday_articles_data = scrape_blog(
        itsjustabadday_base_url,
        itsjustabadday_num_pages,
        extract_itsjustabadday_article_data
    )

    df_itsjustabadday_articles = pd.DataFrame(itsjustabadday_articles_data)
    print("\nDataFrame 'df_itsjustabadday_articles' created successfully.")
    print("Here are the first 5 rows:")
    print(df_itsjustabadday_articles.head())
    print(df_itsjustabadday_articles.info())

    # Scrape full article content for each unique link found on itsjustabadday.com
    unique_itsjustabadday_links = df_itsjustabadday_articles['article_link'].unique().tolist()
    itsjustabadday_full_contents = scrape_article_full_content_itsjustabadday(unique_itsjustabadday_links)
    df_itsjustabadday_content = pd.DataFrame(itsjustabadday_full_contents)

    print(f"\nTotal {len(df_itsjustabadday_content)} article contents extracted for itsjustabadday.com.")
    print("DataFrame 'df_itsjustabadday_content' created successfully. Here are the first 5 rows:")
    print(df_itsjustabadday_content.head())
    print(df_itsjustabadday_content.info())

    # Merge itsjustabadday.com DataFrames to get all information in one place
    df_merged_itsjustabadday = pd.merge(
        df_itsjustabadday_articles,
        df_itsjustabadday_content,
        on='article_link',
        how='left'  # Use a left merge to keep all articles from df_itsjustabadday_articles
    )
    print("\nMerged DataFrame for itsjustabadday.com created successfully.")
    print("Here are the first 5 rows:")
    print(df_merged_itsjustabadday.head())
    print(df_merged_itsjustabadday.info())

    # Save the merged DataFrame to a CSV file
    output_csv_itsjustabadday = 'itsjustabadday_all.csv'
    df_merged_itsjustabadday.to_csv(output_csv_itsjustabadday, index=False)
    print(f"\nMerged DataFrame for itsjustabadday.com saved to {output_csv_itsjustabadday}")


    # --- Scrape psohoward.com blog ---
    print("\n--- Starting scraping for psohoward.com ---")
    psohoward_base_url = 'https://psohoward.com/'
    psohoward_num_pages = 20
    psohoward_start_page = 1 # We'll scrape from page 1 to 20

    # For psohoward.com, the extract_psohoward_article_data function already handles
    # all required fields, including categories, comments_count, likes_count, and article_content,
    # so a separate scrape for full content is not needed.
    psohoward_articles_data = scrape_blog(
        psohoward_base_url,
        psohoward_num_pages,
        extract_psohoward_article_data,
        start_page=psohoward_start_page
    )

    df_psohoward_articles = pd.DataFrame(psohoward_articles_data)
    print("\nDataFrame 'df_psohoward_articles' created successfully.")
    print("Here are the first 5 rows:")
    print(df_psohoward_articles.head())
    print(df_psohoward_articles.info())

    # Save the DataFrame to a CSV file
    output_csv_psohoward = 'psohoward_articles.csv'
    df_psohoward_articles.to_csv(output_csv_psohoward, index=False)
    print(f"\nDataFrame for psohoward.com saved to {output_csv_psohoward}")
