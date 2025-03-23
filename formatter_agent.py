import asyncio
import json
import logging
import re
import os
from openai import AsyncOpenAI, OpenAI
from crawleragent import PropertyListing
from dotenv import load_dotenv
import time
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Add these new models to represent your desired output format
class AddressInfo(BaseModel):
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None

class PropertyLocation(BaseModel):
    lat: Optional[str] = None
    lng: Optional[str] = None

class ListingDetails(BaseModel):
    description: Optional[str] = None
    price: Optional[str] = None
    currency: Optional[str] = None
    status: Optional[str] = None
    listing_type: Optional[str] = None
    category: Optional[str] = None

class FeatureItem(BaseModel):
    feature: str
    value: str

class UserContact(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None

class PropertyOutput(BaseModel):
    address: AddressInfo
    property: PropertyLocation
    listing: ListingDetails
    features: List[FeatureItem] = []
    files: List[str] = []
    user: UserContact

async def format_all_listings(input_file: str) -> str:
    """Format cleaned listings into an official report using OpenAI LLM"""
    logger.info(f"Formatting listings from {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            cleaned_listings = json.load(f)
        
        json_str = json.dumps(cleaned_listings, indent=2)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a formatting assistant. Convert the provided JSON array of real estate listings into a polished, "
                    "official-looking Markdown report. Include a header with title 'Official Property Listings Report', "
                    "generation date, and total listings. For each listing, create a section with: "
                    "title (address or property type), price (formatted with currency), listing type, bedrooms, bathrooms, "
                    "square footage, description, amenities (as a list), image link, and additional info (as a table). "
                    "Return the full Markdown text."
                )},
                {"role": "user", "content": f"Format this JSON into a Markdown report:\n\n{json_str}"}
            ]
        )
        markdown_text = response.choices[0].message.content.strip()
        if markdown_text.startswith("```markdown"):
            markdown_text = re.sub(r'```markdown\s*|\s*```', '', markdown_text)
        logger.info(f"Formatted {len(cleaned_listings)} listings")
        return markdown_text
    except Exception as e:
        logger.error(f"Failed to format {input_file}: {str(e)}")
        return "# Error\n\nFormatting failed."

def format_all_listings_fully_sync(input_file: str) -> str:
    """Fully synchronous version of format_all_listings without asyncio"""
    logger.info(f"Formatting listings from {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            cleaned_listings = json.load(f)
        
        # Use synchronous OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        json_str = json.dumps(cleaned_listings, indent=2)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are a formatting assistant. Convert the provided JSON array of real estate listings into a polished, "
                    "official-looking Markdown report. Include a header with title 'Official Property Listings Report', "
                    "generation date, and total listings. For each listing, create a section with: "
                    "title (address or property type), price (formatted with currency), listing type, bedrooms, bathrooms, "
                    "square footage, description, amenities (as a list), image link, and additional info (as a table). "
                    "Return the full Markdown text."
                )},
                {"role": "user", "content": f"Format this JSON into a Markdown report:\n\n{json_str}"}
            ]
        )
        markdown_text = response.choices[0].message.content.strip()
        if markdown_text.startswith("```markdown"):
            markdown_text = re.sub(r'```markdown\s*|\s*```', '', markdown_text)
        logger.info(f"Formatted {len(cleaned_listings)} listings")
        return markdown_text
    except Exception as e:
        logger.error(f"Failed to format {input_file}: {str(e)}")
        return "# Error\n\nFormatting failed."

def format_all_listings_sync(property_listings: List[PropertyListing], output_dir: str = "output") -> dict:
    """Format all property listings and save as Markdown and JSON."""
    print(f"Formatting {len(property_listings)} listings...")
    
    # Generate and save the markdown report
    md_path = create_markdown_report(property_listings, output_dir)
    print(f"Markdown report saved to: {md_path}")
    
    # Generate and save the JSON in the new format
    json_path = save_json_report(property_listings, output_dir)
    print(f"JSON data saved to: {json_path}")
    
    return {
        "markdown_path": md_path,
        "json_path": json_path,
        "listing_count": len(property_listings)
    }

def transform_to_output_format(property_listing: PropertyListing) -> PropertyOutput:
    """Transform a PropertyListing to the desired output format."""
    # Create address info
    address = AddressInfo(
        country=property_listing.country,
        region=property_listing.region,
        city=property_listing.city,
        district=property_listing.district
    )
    
    # Create property location
    property_location = PropertyLocation(
        lat=property_listing.latitude,
        lng=property_listing.longitude
    )
    
    # Create listing details
    listing_details = ListingDetails(
        description=property_listing.description,
        price=str(property_listing.price) if property_listing.price is not None else None,
        currency=property_listing.currency,
        status=property_listing.status,
        listing_type=property_listing.listing_type,
        category=property_listing.property_type
    )
    
    # Create feature items
    features = []
    
    # Handle specific features
    if property_listing.bedrooms is not None:
        features.append(FeatureItem(feature="Bedroom", value=str(property_listing.bedrooms)))
    
    if property_listing.bathrooms is not None:
        features.append(FeatureItem(feature="Bathroom", value=str(property_listing.bathrooms)))
    
    if property_listing.land_size is not None:
        features.append(FeatureItem(feature="Land size", value=str(property_listing.land_size)))
    
    if property_listing.building_size is not None:
        features.append(FeatureItem(feature="Building size", value=str(property_listing.building_size)))
    
    if property_listing.parking_spaces is not None:
        features.append(FeatureItem(feature="Parking spaces", value=str(property_listing.parking_spaces)))
    
    # Handle boolean features
    if property_listing.has_pool is not None:
        features.append(FeatureItem(feature="Pool", value=str(property_listing.has_pool).lower()))
        features.append(FeatureItem(feature="Swimming pool", value=str(property_listing.has_pool).lower()))
    
    if property_listing.has_garage is not None:
        features.append(FeatureItem(feature="Remote garage", value=str(property_listing.has_garage).lower()))
    
    if property_listing.has_ac is not None:
        features.append(FeatureItem(feature="Airconditioning", value=str(property_listing.has_ac).lower()))
    
    # Additional features from amenities
    if property_listing.amenities:
        for amenity in property_listing.amenities:
            if "pool" in amenity.lower() and "outdoor" in amenity.lower():
                features.append(FeatureItem(feature="Outdoor swimming pool", value="true"))
    
    # Create user contact info
    user_contact = UserContact(
        email=property_listing.contact_email,
        phone=property_listing.contact_phone
    )
    
    # Prepare image URLs
    image_urls = property_listing.image_urls if property_listing.image_urls else []
    
    # Create the final output object
    return PropertyOutput(
        address=address,
        property=property_location,
        listing=listing_details,
        features=features,
        files=image_urls,
        user=user_contact
    )

def save_json_report(property_listings: List[PropertyListing], output_dir: str = "output") -> str:
    """Save property listings in the desired JSON format."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Transform listings to the output format
    output_listings = [transform_to_output_format(listing).dict() for listing in property_listings]
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"property_listings_{timestamp}.json")
    
    # Write JSON file with 2-space indentation
    with open(json_path, "w") as f:
        json.dump(output_listings, f, indent=2)
    
    return json_path

def create_markdown_report(property_listings: List[PropertyListing], output_dir: str = "output") -> str:
    """Create and save a markdown report from property listings."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"property_report_{timestamp}.md")
    
    # Create markdown content
    markdown = "# Official Property Listings Report\n\n"
    markdown += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += f"Total Listings: {len(property_listings)}\n\n"
    
    for idx, listing in enumerate(property_listings, 1):
        markdown += f"## Listing {idx}: {listing.address or listing.property_type}\n\n"
        if listing.price is not None and listing.currency:
            markdown += f"**Price:** {listing.currency} {listing.price:,}\n\n"
        if listing.listing_type:
            markdown += f"**Listing Type:** {listing.listing_type}\n\n"
        
        # Property details
        markdown += "### Property Details\n\n"
        if listing.bedrooms is not None:
            markdown += f"- **Bedrooms:** {listing.bedrooms}\n"
        if listing.bathrooms is not None:
            markdown += f"- **Bathrooms:** {listing.bathrooms}\n"
        if listing.building_size is not None:
            markdown += f"- **Building Size:** {listing.building_size} m²\n"
        if listing.land_size is not None:
            markdown += f"- **Land Size:** {listing.land_size} m²\n"
        markdown += "\n"
        
        # Description
        if listing.description:
            markdown += "### Description\n\n"
            markdown += f"{listing.description}\n\n"
        
        # Amenities
        if listing.amenities and len(listing.amenities) > 0:
            markdown += "### Amenities\n\n"
            for amenity in listing.amenities:
                markdown += f"- {amenity}\n"
            markdown += "\n"
        
        # Images
        if listing.image_urls and len(listing.image_urls) > 0:
            markdown += "### Images\n\n"
            for img_url in listing.image_urls:
                markdown += f"- [{img_url}]({img_url})\n"
            markdown += "\n"
        
        # Contact info
        markdown += "### Contact Information\n\n"
        if listing.contact_email:
            markdown += f"- **Email:** {listing.contact_email}\n"
        if listing.contact_phone:
            markdown += f"- **Phone:** {listing.contact_phone}\n"
        markdown += "\n"
        
        # Separator between listings
        if idx < len(property_listings):
            markdown += "---\n\n"
    
    # Write markdown to file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    return md_path

async def format_listings_async(property_listings: List[PropertyListing], output_dir: str = "output") -> dict:
    """Format property listings asynchronously and save as Markdown and JSON."""
    print(f"Formatting {len(property_listings)} listings asynchronously...")
    
    # Create markdown report
    md_path = create_markdown_report(property_listings, output_dir)
    print(f"Markdown report saved to: {md_path}")
    
    # Save JSON in the new format
    json_path = save_json_report(property_listings, output_dir)
    print(f"JSON data saved to: {json_path}")
    
    return {
        "markdown_path": md_path,
        "json_path": json_path,
        "listing_count": len(property_listings)
    }

def format_all_listings_from_file_sync(input_file: str, output_dir: str = "output") -> dict:
    """Format all listings from a JSON file."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert JSON data to PropertyListing objects
        property_listings = []
        for item in data:
            try:
                listing = PropertyListing.parse_obj(item)
                property_listings.append(listing)
            except Exception as e:
                logger.error(f"Failed to parse listing: {str(e)}")
        
        return format_all_listings_sync(property_listings, output_dir)
    except Exception as e:
        logger.error(f"Failed to format listings from file: {str(e)}")
        return {
            "error": str(e),
            "listing_count": 0
        }

async def format_all_listings_from_file(input_file: str, output_dir: str = "output") -> dict:
    """Format all listings from a JSON file asynchronously."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert JSON data to PropertyListing objects
        property_listings = []
        for item in data:
            try:
                listing = PropertyListing.parse_obj(item)
                property_listings.append(listing)
            except Exception as e:
                logger.error(f"Failed to parse listing: {str(e)}")
        
        return await format_listings_async(property_listings, output_dir)
    except Exception as e:
        logger.error(f"Failed to format listings from file: {str(e)}")
        return {
            "error": str(e),
            "listing_count": 0
        }

def format_all_listings_from_file_fully_sync(input_file: str, output_dir: str = "output") -> dict:
    """Fully synchronous version of formatting from a file."""
    return format_all_listings_from_file_sync(input_file, output_dir)

# Wrapper functions to support both sync and async
def format_listings(property_listings: List[PropertyListing], output_dir: str = "output", use_sync: bool = False) -> dict:
    """Format property listings and handle sync/async modes."""
    if use_sync:
        return format_all_listings_sync(property_listings, output_dir)
    else:
        return asyncio.run(format_listings_async(property_listings, output_dir))

def format_from_file(input_file: str, output_dir: str = "output", use_sync: bool = False) -> dict:
    """Format listings from a file and handle sync/async modes."""
    if use_sync:
        return format_all_listings_from_file_sync(input_file, output_dir)
    else:
        return asyncio.run(format_all_listings_from_file(input_file, output_dir))

# Command-line functionality if needed
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Format property listings")
    parser.add_argument("input_file", help="Path to the cleaned JSON file")
    parser.add_argument("--output-dir", default="output", help="Directory to save output")
    parser.add_argument("--sync", action="store_true", help="Use synchronous processing")
    args = parser.parse_args()
    
    result = format_from_file(args.input_file, args.output_dir, args.sync)
    print(f"Formatted {result.get('listing_count', 0)} listings")
    if "markdown_path" in result:
        print(f"Markdown saved to: {result['markdown_path']}")
    if "json_path" in result:
        print(f"JSON saved to: {result['json_path']}")