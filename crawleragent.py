import asyncio
import os
import re
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from crawl4ai import AsyncWebCrawler
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from openai import AsyncOpenAI, OpenAI
import tiktoken
from pydantic import BaseModel, Field, field_validator
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set up OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Set ProactorEventLoop for Windows compatibility with Playwright
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class PropertyListing(BaseModel):
    address: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    square_footage: Optional[int] = None
    property_type: Optional[str] = None
    listing_type: Optional[str] = None
    year_built: Optional[int] = None
    description: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    source: Optional[str] = None
    listing_date: Optional[str] = None
    image_link: Optional[str] = None
    additional_info: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('price', mode='before')
    def validate_price(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = re.sub(r'[^\d.]', '', v)
            return float(cleaned) if cleaned else None
        return None

    @field_validator('currency', mode='before')
    def validate_currency(cls, v, info):
        values = info.data
        if v is not None:
            v = v.upper()
            valid_currencies = {'USD', 'MXN', 'EUR', 'CAD'}
            if v in valid_currencies:
                return v
        price_str = values.get('price', '') if isinstance(values.get('price'), str) else ''
        if isinstance(price_str, str):
            if '$' in price_str and 'MXN' not in price_str:
                return 'USD'
            elif 'MXN' in price_str or '$' in price_str:
                return 'MXN'
        return 'MXN'

    @field_validator('listing_type')
    def validate_listing_type(cls, v):
        if v is None:
            return None
        v = v.lower()
        if v in ['rent', 'buy', 'sale', 'rental']:
            return 'rent' if v in ['rent', 'rental'] else 'buy'
        return None

    @field_validator('bedrooms', 'bathrooms', mode='before')
    def validate_rooms(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            match = re.search(r'(\d+(\.\d+)?)', v)
            return float(match.group(1)) if match else None
        return None

    @field_validator('square_footage', mode='before')
    def validate_square_footage(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            match = re.search(r'(\d+)', v.replace(',', ''))
            return int(match.group(1)) if match else None
        return None

    @field_validator('year_built', mode='before')
    def validate_year_built(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            match = re.search(r'(\d{4})', v)
            if match:
                year = int(match.group(1))
                current_year = datetime.now().year
                if 1800 <= year <= current_year:
                    return year
            return None
        return None

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    return ' '.join(text.split())

def create_smart_chunks(text, max_tokens=500, overlap_tokens=50):
    cleaned_text = clean_text(text)
    listing_patterns = [
        r'\$[\d,]+', r'\d+ bed', r'\d+ bath', r'\d+\s+sq\s*\.?\s*ft',
        r'\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)'
    ]
    boundaries = [0]
    for pattern in listing_patterns:
        for match in re.finditer(pattern, cleaned_text, re.IGNORECASE):
            sentence_start = cleaned_text.rfind('.', 0, match.start())
            sentence_start = 0 if sentence_start == -1 else sentence_start + 1
            if sentence_start not in boundaries:
                boundaries.append(sentence_start)
    boundaries.sort()
    
    chunks = []
    current_start = 0
    current_tokens = 0
    
    for i, boundary in enumerate(boundaries[1:], 1):
        segment = cleaned_text[current_start:boundary].strip()
        segment_tokens = len(tokenizer.encode(segment))
        
        if current_tokens + segment_tokens > max_tokens:
            chunk_end = boundaries[i-1]
            chunks.append(cleaned_text[current_start:chunk_end])
            overlap_start = max(0, chunk_end - overlap_tokens)
            current_start = overlap_start
            current_tokens = len(tokenizer.encode(cleaned_text[overlap_start:boundary]))
        current_tokens += segment_tokens
    
    if current_start < len(cleaned_text):
        chunks.append(cleaned_text[current_start:])
    return chunks

async def extract_housing_info(text, max_chunks=None, delay_seconds=1.0, max_total_tokens=None):
    chunks = create_smart_chunks(text)
    
    # Limit number of chunks if specified
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]
    
    # Apply max_total_tokens limit if specified
    if max_total_tokens is not None and max_total_tokens > 0:
        total_tokens = 0
        limited_chunks = []
        for chunk in chunks:
            chunk_tokens = len(tokenizer.encode(chunk))
            if total_tokens + chunk_tokens <= max_total_tokens:
                limited_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        chunks = limited_chunks
        
    all_housing_info = []
    print(f"Processing {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        try:
            # Add delay between requests to avoid rate limiting
            if i > 0 and delay_seconds > 0:
                print(f"Waiting {delay_seconds} seconds before processing next chunk...")
                await asyncio.sleep(delay_seconds)
            
            # Calculate approximate token count for logging/monitoring
            chunk_tokens = len(tokenizer.encode(chunk))
            print(f"Chunk {i+1} size: ~{chunk_tokens} tokens")
                
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": (
                        "You are a strict and precise assistant tasked with extracting reliable real estate listings from text. "
                        "Each listing must be a complete, legitimate property with structured details. "
                        "Incomplete or unclear listings must be discarded. "
                        "Required fields: address, price, currency, bedrooms, bathrooms, listing_type (rent or buy), "
                        "property_type, description, image_link. Optional: square_footage, year_built, amenities, additional_info. "
                        "Return a JSON array of valid listings."
                    )},
                    {"role": "user", "content": (
                        f"Extract real estate listings from the text below in JSON format. "
                        f"Required fields:\n"
                        f"- `address` (string)\n"
                        f"- `price` (float)\n"
                        f"- `currency` (string, e.g., 'USD', 'MXN')\n"
                        f"- `bedrooms` (float)\n"
                        f"- `bathrooms` (float)\n"
                        f"- `listing_type` (string, 'rent' or 'buy')\n"
                        f"- `property_type` (string, e.g., 'house')\n"
                        f"- `description` (string)\n"
                        f"- `image_link` (string, valid URL)\n"
                        f"Optional fields:\n"
                        f"- `square_footage` (integer or null)\n"
                        f"- `year_built` (integer or null)\n"
                        f"- `amenities` (array of strings)\n"
                        f"- `additional_info` (object)\n"
                        f"Discard incomplete listings. Infer currency if needed (e.g., MXN for Mexico). Text:\n\n{chunk}"
                    )}
                ]
            )
            response_text = response.choices[0].message.content
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)
            
            chunk_info = json.loads(response_text)
            listings = chunk_info if isinstance(chunk_info, list) else [chunk_info]
            
            for listing_data in listings:
                if isinstance(listing_data, dict):
                    if 'source' not in listing_data:
                        listing_data['source'] = 'Casas y Terrenos'
                    validated_listing = PropertyListing(**listing_data)
                    all_housing_info.append(validated_listing)
                    print(f"Validated listing: {validated_listing.address or 'Unknown address'}")
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
    return all_housing_info

async def scrape_with_playwright_crawl4ai(url, max_depth, max_pages):
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=max_depth,
            include_external=False,
            max_pages=max_pages
        ),
        verbose=True
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(
            url=url,
            config=run_config
        )
        return results  # Returns a list of CrawlResult objects

async def process_scraping(url, max_depth, max_pages, max_chunks=None, delay_seconds=1.0, max_total_tokens=None):
    print(f"Scraping {url} with max_depth={max_depth}, max_pages={max_pages}...")
    results = await scrape_with_playwright_crawl4ai(url, max_depth, max_pages)
    
    # Combine content from all successful results
    combined_content = ""
    success = False
    for result in results:
        if result.success:
            combined_content += str(result) + "\n"
            success = True
    
    if not success:
        print(f"Failed to scrape {url}")
        return 0
    
    print("Scraped content successfully")
    housing_info = await extract_housing_info(
        combined_content, 
        max_chunks=max_chunks, 
        delay_seconds=delay_seconds,
        max_total_tokens=max_total_tokens
    )
    print(f"Extracted {len(housing_info)} listings")
    
    output_file = f"raw_property_listings_{url_hash(url)}.json"
    json_data = [listing.dict() for listing in housing_info]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved raw data to {output_file}")
    return len(housing_info)

def process_scraping_sync(url, max_depth, max_pages, max_chunks=None, delay_seconds=1.0, max_total_tokens=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_scraping(
        url, max_depth, max_pages, max_chunks, delay_seconds, max_total_tokens
    ))
    loop.close()
    return result

def url_hash(url):
    return re.sub(r'[^a-zA-Z0-9]', '_', url)[:50]

def run_gui():
    from property_pipeline import queue_tasks

    root = tk.Tk()
    root.title("Property Scraper")
    root.geometry("800x750")
    root.configure(bg="#2E2E2E")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#2E2E2E", foreground="#FFFFFF", font=("Helvetica", 12))
    style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10, background="#4CAF50", foreground="#FFFFFF")
    style.map("TButton", background=[("active", "#45A049")])
    style.configure("TEntry", fieldbackground="#424242", foreground="#FFFFFF", font=("Helvetica", 11))
    style.configure("TCheckbutton", background="#2E2E2E", foreground="#FFFFFF", font=("Helvetica", 11))
    style.map("TCheckbutton", background=[("active", "#2E2E2E")])

    ttk.Label(root, text="Property Web Scraper", font=("Helvetica", 18, "bold"), foreground="#4CAF50").pack(pady=15)

    frame = ttk.Frame(root, padding="20")
    frame.pack(fill="both", expand=True)

    # URL Input
    ttk.Label(frame, text="Websites to Scrape (one per line):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    url_text = tk.Text(frame, height=5, width=60, bg="#424242", fg="#FFFFFF", insertbackground="white")
    url_text.grid(row=1, column=0, columnspan=2, padx=10, pady=5)
    url_text.insert("1.0", "https://www.casasyterrenos.com/jalisco/Puerto%20Vallarta/casas/venta?desde=0&hasta=5000000")

    # Crawler Configuration
    crawler_frame = ttk.LabelFrame(frame, text="Crawler Settings", padding=10)
    crawler_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="we")
    
    ttk.Label(crawler_frame, text="Max Depth:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    depth_entry = ttk.Entry(crawler_frame, width=10)
    depth_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    depth_entry.insert(0, "2")

    ttk.Label(crawler_frame, text="Max Pages:").grid(row=0, column=2, padx=10, pady=5, sticky="e")
    pages_entry = ttk.Entry(crawler_frame, width=10)
    pages_entry.grid(row=0, column=3, padx=10, pady=5, sticky="w")
    pages_entry.insert(0, "5")
    
    # API Rate Limiting Configuration
    api_frame = ttk.LabelFrame(frame, text="API Rate Limiting", padding=10)
    api_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="we")
    
    ttk.Label(api_frame, text="Max Chunks:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    chunks_entry = ttk.Entry(api_frame, width=10)
    chunks_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    chunks_entry.insert(0, "3")
    
    ttk.Label(api_frame, text="Max Total Tokens:").grid(row=0, column=2, padx=10, pady=5, sticky="e")
    tokens_entry = ttk.Entry(api_frame, width=10)
    tokens_entry.grid(row=0, column=3, padx=10, pady=5, sticky="w")
    tokens_entry.insert(0, "10000")
    
    ttk.Label(api_frame, text="Scraper Delay (seconds):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    delay_entry = ttk.Entry(api_frame, width=10)
    delay_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    delay_entry.insert(0, "2.0")
    
    ttk.Label(api_frame, text="Cleaner Delay (seconds):").grid(row=1, column=2, padx=10, pady=5, sticky="e")
    cleaner_delay_entry = ttk.Entry(api_frame, width=10)
    cleaner_delay_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")
    cleaner_delay_entry.insert(0, "1.0")
    
    # API Usage Warning
    warning_frame = ttk.LabelFrame(frame, text="API Usage Warning", padding=10)
    warning_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="we")
    
    warning_text = (
        "Note: Each chunk consumes approximately 500-1000 tokens.\n"
        "Smaller chunks and longer delays help avoid API rate limits.\n"
        "Recommended: max 3-5 chunks, 2+ second delays between requests."
    )
    warning_label = ttk.Label(warning_frame, text=warning_text, foreground="#FFA500")
    warning_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
    
    # Advanced Options
    advanced_frame = ttk.LabelFrame(frame, text="Advanced Options", padding=10)
    advanced_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="we")
    
    use_sync_var = tk.BooleanVar(value=True)
    sync_checkbox = ttk.Checkbutton(
        advanced_frame, 
        text="Use fully synchronous mode (no asyncio)", 
        variable=use_sync_var
    )
    sync_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    # Status and Submit
    status_label = ttk.Label(frame, text="Ready to scrape", foreground="#FFFFFF", font=("Helvetica", 11, "italic"))
    status_label.grid(row=6, column=0, columnspan=2, pady=10)

    submit_button = ttk.Button(frame, text="Start Scraping", command=lambda: on_submit())
    submit_button.grid(row=7, column=0, columnspan=2, pady=20)

    def on_submit():
        urls = [url.strip() for url in url_text.get("1.0", tk.END).splitlines() if url.strip()]
        try:
            max_depth = int(depth_entry.get().strip())
            max_pages = int(pages_entry.get().strip())
            max_chunks_val = int(chunks_entry.get().strip())
            max_tokens = int(tokens_entry.get().strip())
            delay_seconds = float(delay_entry.get().strip())
            cleaner_delay = float(cleaner_delay_entry.get().strip())
            use_sync = use_sync_var.get()
            
            # Validate inputs
            if not urls or max_depth < 0 or max_pages < 1 or delay_seconds < 0 or cleaner_delay < 0:
                raise ValueError
            
            if max_chunks_val < 1:
                messagebox.showwarning("Warning", "Setting max chunks to at least 1 to ensure some results are processed.")
                max_chunks_val = 1
                
            if max_tokens < 1000:
                messagebox.showwarning("Warning", "Setting max tokens to at least 1000 to ensure one chunk can be processed.")
                max_tokens = 1000
                
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid URLs, non-negative depth and delays, and pages >= 1.")
            return
        
        # Set environment variable for synchronous mode
        if use_sync:
            os.environ["USE_SYNC"] = "true"
            status_label.config(text=f"Running in synchronous mode. Queuing {len(urls)} websites...", foreground="#FFA500")
        else:
            os.environ["USE_SYNC"] = "false"
            status_label.config(text=f"Queuing {len(urls)} websites...", foreground="#FFA500")
            
        submit_button.config(state="disabled")
        
        def queue_and_update():
            queue_tasks(
                urls, 
                max_depth, 
                max_pages, 
                max_chunks_val, 
                delay_seconds, 
                cleaner_delay,
                max_tokens
            )
            root.after(0, lambda: status_label.config(text=f"Queued {len(urls)} websites!", foreground="green"))
            root.after(0, lambda: submit_button.config(state="normal"))

        thread = threading.Thread(target=queue_and_update)
        thread.start()

    root.mainloop()

def extract_housing_info_sync(text, max_chunks=None, delay_seconds=1.0, max_total_tokens=None):
    """Synchronous version of extract_housing_info"""
    chunks = create_smart_chunks(text)
    
    # Limit number of chunks if specified
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]
    
    # Apply max_total_tokens limit if specified
    if max_total_tokens is not None and max_total_tokens > 0:
        total_tokens = 0
        limited_chunks = []
        for chunk in chunks:
            chunk_tokens = len(tokenizer.encode(chunk))
            if total_tokens + chunk_tokens <= max_total_tokens:
                limited_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        chunks = limited_chunks
        
    all_housing_info = []
    print(f"Processing {len(chunks)} chunks")

    # Use synchronous OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    for i, chunk in enumerate(chunks):
        try:
            # Add delay between requests to avoid rate limiting
            if i > 0 and delay_seconds > 0:
                print(f"Waiting {delay_seconds} seconds before processing next chunk...")
                time.sleep(delay_seconds)
            
            # Calculate approximate token count for logging/monitoring
            chunk_tokens = len(tokenizer.encode(chunk))
            print(f"Chunk {i+1} size: ~{chunk_tokens} tokens")
                
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": (
                        "You are a strict and precise assistant tasked with extracting reliable real estate listings from text. "
                        "Each listing must be a complete, legitimate property with structured details. "
                        "Incomplete or unclear listings must be discarded. "
                        "Required fields: address, price, currency, bedrooms, bathrooms, listing_type (rent or buy), "
                        "property_type, description, image_link. Optional: square_footage, year_built, amenities, additional_info. "
                        "Return a JSON array of valid listings."
                    )},
                    {"role": "user", "content": (
                        f"Extract real estate listings from the text below in JSON format. "
                        f"Required fields:\n"
                        f"- `address` (string)\n"
                        f"- `price` (float)\n"
                        f"- `currency` (string, e.g., 'USD', 'MXN')\n"
                        f"- `bedrooms` (float)\n"
                        f"- `bathrooms` (float)\n"
                        f"- `listing_type` (string, 'rent' or 'buy')\n"
                        f"- `property_type` (string, e.g., 'house')\n"
                        f"- `description` (string)\n"
                        f"- `image_link` (string, valid URL)\n"
                        f"Optional fields:\n"
                        f"- `square_footage` (integer or null)\n"
                        f"- `year_built` (integer or null)\n"
                        f"- `amenities` (array of strings)\n"
                        f"- `additional_info` (object)\n"
                        f"Discard incomplete listings. Infer currency if needed (e.g., MXN for Mexico). Text:\n\n{chunk}"
                    )}
                ]
            )
            response_text = response.choices[0].message.content
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)
            
            chunk_info = json.loads(response_text)
            listings = chunk_info if isinstance(chunk_info, list) else [chunk_info]
            
            for listing_data in listings:
                if isinstance(listing_data, dict):
                    if 'source' not in listing_data:
                        listing_data['source'] = 'Casas y Terrenos'
                    validated_listing = PropertyListing(**listing_data)
                    all_housing_info.append(validated_listing)
                    print(f"Validated listing: {validated_listing.address or 'Unknown address'}")
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
    return all_housing_info

def scrape_with_playwright_crawl4ai_sync(url, max_depth, max_pages):
    """Synchronous version of scrape_with_playwright_crawl4ai"""
    from crawl4ai import WebCrawler
    from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
    from crawl4ai.configs import BrowserConfig, CrawlerRunConfig
    
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=max_depth,
            include_external=False,
            max_pages=max_pages
        ),
        verbose=True
    )
    
    with WebCrawler(config=browser_config) as crawler:
        results = crawler.run(
            url=url,
            config=run_config
        )
        return results

def process_scraping_fully_sync(url, max_depth, max_pages, max_chunks=None, delay_seconds=1.0, max_total_tokens=None):
    """Fully synchronous version of process_scraping without asyncio"""
    print(f"Scraping {url} with max_depth={max_depth}, max_pages={max_pages}...")
    results = scrape_with_playwright_crawl4ai_sync(url, max_depth, max_pages)
    
    # Combine content from all successful results
    combined_content = ""
    success = False
    for result in results:
        if result.success:
            combined_content += str(result) + "\n"
            success = True
    
    if not success:
        print(f"Failed to scrape {url}")
        return 0
    
    print("Scraped content successfully")
    housing_info = extract_housing_info_sync(
        combined_content, 
        max_chunks=max_chunks, 
        delay_seconds=delay_seconds,
        max_total_tokens=max_total_tokens
    )
    print(f"Extracted {len(housing_info)} listings")
    
    output_file = f"raw_property_listings_{url_hash(url)}.json"
    json_data = [listing.dict() for listing in housing_info]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved raw data to {output_file}")
    return len(housing_info)

if __name__ == "__main__":
    run_gui()