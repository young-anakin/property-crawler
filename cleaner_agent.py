import asyncio
import json
import logging
import re
import os
import time
from typing import List
from openai import AsyncOpenAI, OpenAI
from crawleragent import PropertyListing
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def clean_all_listings(input_file: str, delay_seconds=0.0) -> List[PropertyListing]:
    """Clean raw listings using OpenAI LLM"""
    logger.info(f"Cleaning listings from {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_listings = json.load(f)
        
        # If there's a request delay specified and we have multiple listings, wait before making the API call
        if delay_seconds > 0 and len(raw_listings) > 0:
            logger.info(f"Waiting {delay_seconds} seconds before API call...")
            await asyncio.sleep(delay_seconds)
        
        raw_json = json.dumps(raw_listings, indent=2)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a data cleaning assistant. Clean and standardize the provided JSON array of real estate listings. "
                    "Ensure all required fields are present and valid: address, price, currency, bedrooms, bathrooms, "
                    "listing_type (rent or buy), property_type, description, image_link. Optional fields: square_footage, "
                    "year_built, amenities, additional_info. Remove incomplete listings (missing required fields). "
                    "Standardize formats (e.g., price as float, remove extra text). Return a cleaned JSON array."
                )},
                {"role": "user", "content": f"Clean this JSON:\n\n{raw_json}"}
            ]
        )
        response_text = response.choices[0].message.content
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        cleaned_data = json.loads(json_match.group(1)) if json_match else json.loads(response_text)
        
        cleaned_listings = []
        for listing_data in cleaned_data:
            try:
                cleaned_listing = PropertyListing(**listing_data)
                cleaned_listings.append(cleaned_listing)
                logger.info(f"Cleaned listing: {cleaned_listing.address or 'Unknown address'}")
            except Exception as e:
                logger.warning(f"Skipping invalid listing: {str(e)}")
        
        logger.info(f"Cleaned {len(cleaned_listings)} listings")
        return cleaned_listings
    except Exception as e:
        logger.error(f"Failed to clean {input_file}: {str(e)}")
        return []

def clean_all_listings_fully_sync(input_file: str, delay_seconds=0.0) -> List[PropertyListing]:
    """Fully synchronous version of clean_all_listings without asyncio"""
    logger.info(f"Cleaning listings from {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_listings = json.load(f)
        
        # If there's a request delay specified and we have multiple listings, wait before making the API call
        if delay_seconds > 0 and len(raw_listings) > 0:
            logger.info(f"Waiting {delay_seconds} seconds before API call...")
            time.sleep(delay_seconds)
        
        # Use synchronous OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        raw_json = json.dumps(raw_listings, indent=2)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a data cleaning assistant. Clean and standardize the provided JSON array of real estate listings. "
                    "Ensure all required fields are present and valid: address, price, currency, bedrooms, bathrooms, "
                    "listing_type (rent or buy), property_type, description, image_link. Optional fields: square_footage, "
                    "year_built, amenities, additional_info. Remove incomplete listings (missing required fields). "
                    "Standardize formats (e.g., price as float, remove extra text). Return a cleaned JSON array."
                )},
                {"role": "user", "content": f"Clean this JSON:\n\n{raw_json}"}
            ]
        )
        response_text = response.choices[0].message.content
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        cleaned_data = json.loads(json_match.group(1)) if json_match else json.loads(response_text)
        
        cleaned_listings = []
        for listing_data in cleaned_data:
            try:
                cleaned_listing = PropertyListing(**listing_data)
                cleaned_listings.append(cleaned_listing)
                logger.info(f"Cleaned listing: {cleaned_listing.address or 'Unknown address'}")
            except Exception as e:
                logger.warning(f"Skipping invalid listing: {str(e)}")
        
        logger.info(f"Cleaned {len(cleaned_listings)} listings")
        return cleaned_listings
    except Exception as e:
        logger.error(f"Failed to clean {input_file}: {str(e)}")
        return []

def clean_all_listings_sync(input_file: str, output_file: str, delay_seconds=0.0) -> int:
    """Synchronous wrapper for cleaning"""
    # Check if we should use the new fully synchronous version
    use_sync = os.getenv("USE_SYNC", "false").lower() == "true"
    
    if use_sync:
        # Use fully synchronous version
        cleaned_listings = clean_all_listings_fully_sync(input_file, delay_seconds)
    else:
        # Use async version with event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cleaned_listings = loop.run_until_complete(clean_all_listings(input_file, delay_seconds))
        loop.close()
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([listing.dict() for listing in cleaned_listings], f, indent=2)
    logger.info(f"Saved cleaned data to {output_file}")
    return len(cleaned_listings)

if __name__ == "__main__":
    clean_all_listings_sync("raw_property_listings_test.json", "cleaned_property_listings_test.json")