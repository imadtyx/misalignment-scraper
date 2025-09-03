#!/usr/bin/env python3
"""
Main scraper for misalignment detection project.
Uses modular structure with sites and modalities for extensible scraping.

Features:
- Modular architecture with sites/ and modalities/
- Support for multiple platforms (Twitter, future: Reddit, etc.)
- Multiple scraping modes (account, tweet, batch)
- Integrated image processing and analysis
- Comprehensive output formats
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the scrapers directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sites.twitter import TwitterScraper, scrape_twitter_account, scrape_twitter_tweet
from GeminiScraper import GeminiScraper
from modalities.image import ImageProcessor, process_single_image, process_images_from_directory

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def print_banner():
    """Print the application banner."""
    print("=" * 80)
    print("🔍 MISALIGNMENT SCRAPER")
    print("🏗️  Modular Architecture | 🔒 API Safe | 📊 Multi-format Output")
    print("=" * 80)


def scrape_twitter_content(args) -> List[Dict[str, Any]]:
    """
    Scrape Twitter content based on the provided arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of scraped content
    """
    bearer_token = args.bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        raise ValueError("Twitter Bearer Token required. Use --bearer-token or set TWITTER_BEARER_TOKEN env var")
    
    scraper = TwitterScraper(
        bearer_token=bearer_token,
        download_media=args.download_media,
        media_dir=args.media_dir
    )
    
    scraped_data = []
    
    # Handle different scraping modes
    if args.mode == 'account':
        print(f"📱 Scraping Twitter account: @{args.target}")
        data = scraper.scrape_user_tweets(args.target, limit=args.limit)
        scraped_data.extend(data)
        
    elif args.mode == 'tweet':
        print(f"🐦 Scraping individual tweet: {args.target}")
        data = scraper.scrape_tweet(args.target)
        if data:
            scraped_data.append(data)
        
    elif args.mode == 'batch':
        print(f"📋 Batch scraping from file: {args.target}")
        try:
            with open(args.target, 'r') as f:
                targets = [line.strip() for line in f if line.strip()]
            
            for target in targets:
                print(f"Processing: {target}")
                
                # Determine if it's a username or tweet URL/ID
                if target.startswith('@') or (not target.startswith('http') and target.isdigit()):
                    # It's a username or tweet ID
                    if target.startswith('@'):
                        data = scraper.scrape_user_tweets(target, limit=args.limit)
                        scraped_data.extend(data)
                    else:
                        data = scraper.scrape_tweet(target)
                        if data:
                            scraped_data.append(data)
                else:
                    # It's likely a tweet URL
                    data = scraper.scrape_tweet(target)
                    if data:
                        scraped_data.append(data)
                        
        except FileNotFoundError:
            print(f"❌ Batch file not found: {args.target}")
            return []
        except Exception as e:
            print(f"❌ Error processing batch file: {e}")
            return []
    
    # Save the scraped data
    if scraped_data:
        scraper.save_data(scraped_data, args.output_prefix)
    
    return scraped_data


def process_scraped_images(scraped_data: List[Dict[str, Any]], args) -> Optional[List[Dict[str, Any]]]:
    """
    Process images from scraped data using the image modality processor.
    
    Args:
        scraped_data: List of scraped content
        args: Command line arguments
        
    Returns:
        List of image processing results or None if no images to process
    """
    if not args.process_images:
        return None
    
    # Collect all image paths from scraped data
    image_paths = []
    for item in scraped_data:
        if item.get('has_media') and item.get('multimedia', {}).get('items'):
            for media_item in item['multimedia']['items']:
                if media_item.get('modality') == 'image' and media_item.get('local_path'):
                    image_paths.append(media_item['local_path'])
    
    if not image_paths:
        print("ℹ️  No images found in scraped data to process")
        return None
    
    print(f"🖼️  Processing {len(image_paths)} images...")
    
    # Create image processor
    image_processor = ImageProcessor(output_dir=args.image_output_dir)
    
    # Process images
    results = image_processor.process_images_batch(
        image_paths,
        create_thumbnails=args.create_thumbnails,
        extract_text=args.extract_text
    )
    
    # Save results
    if results:
        image_processor.save_results(results, args.output_prefix + "_images")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="🔍 Misalignment Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape Twitter account
  %(prog)s twitter account @elonmusk --limit 5
  
  # Scrape specific tweet
  %(prog)s twitter tweet https://twitter.com/elonmusk/status/1234567890
  
  # Scrape with image processing
  %(prog)s twitter account @elonmusk --limit 5 --download-media --process-images
  
  # Batch scraping from file
  %(prog)s twitter batch targets.txt --limit 3
        """
    )
    
    # Platform selection
    parser.add_argument(
        'platform',
        choices=['twitter', 'gemini'],
        help='Platform to scrape (twitter | gemini)'
    )
    
    # Scraping mode
    parser.add_argument(
        'mode',
        choices=['account', 'tweet', 'batch', 'chat'],
        help='Scraping mode: twitter=[account|tweet|batch], gemini=[chat]'
    )
    
    # Target specification
    parser.add_argument(
        'target',
        help='Target to scrape: username (@user), tweet ID/URL, or batch file path'
    )
    
    # Twitter-specific options
    parser.add_argument(
        '--bearer-token',
        help='Twitter API Bearer Token (or set TWITTER_BEARER_TOKEN env var)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Number of tweets to scrape per account (default: 5, max: 10)'
    )
    
    # Media handling
    parser.add_argument(
        '--download-media',
        action='store_true',
        help='Download media files (images, videos)'
    )
    
    parser.add_argument(
        '--media-dir',
        default='media',
        help='Directory to store downloaded media (default: media)'
    )
    
    # Image processing options
    parser.add_argument(
        '--process-images',
        action='store_true',
        help='Process downloaded images for analysis'
    )
    
    parser.add_argument(
        '--create-thumbnails',
        action='store_true',
        default=True,
        help='Create thumbnails for processed images (default: True)'
    )
    
    parser.add_argument(
        '--extract-text',
        action='store_true',
        help='Extract text from images (requires OCR setup)'
    )
    
    parser.add_argument(
        '--image-output-dir',
        default='processed_images',
        help='Directory for image processing output (default: processed_images)'
    )
    
    # Output options
    parser.add_argument(
        '--output-prefix',
        default='scraped_content',
        help='Prefix for output files (default: scraped_content)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Print banner unless quiet mode
    if not args.quiet:
        print_banner()
    
    try:
        # Platform-specific scraping
        if args.platform == 'twitter':
            scraped_data = scrape_twitter_content(args)
        elif args.platform == 'gemini':
            if args.mode != 'chat':
                print("❌ Gemini only supports mode 'chat' (shared chat URL)")
                return 1
            url = args.target
            print(f"🧪 Scraping Gemini shared chat: {url}")
            gs = GeminiScraper(headless=True)
            transcript = gs.scrape(url)
            out_prefix = args.output_prefix or 'gemini_chat'
            out_path = f"{out_prefix}.json"
            GeminiScraper.save_transcript(transcript, out_path)
            print(f"✅ Saved Gemini transcript to {out_path}")
            scraped_data = [transcript.to_dict()]
        else:
            print(f"❌ Platform '{args.platform}' not yet implemented")
            return 1
        
        if not scraped_data:
            print("❌ No data scraped")
            return 1
        
        # Process images if requested
        image_results = process_scraped_images(scraped_data, args)
        
        # Print summary
        if not args.quiet:
            print(f"\n🎉 SCRAPING COMPLETED SUCCESSFULLY! 🎉")
            print(f"📊 Summary:")
            print(f"   • Platform: {args.platform}")
            print(f"   • Mode: {args.mode}")
            print(f"   • Target: {args.target}")
            print(f"   • Items scraped: {len(scraped_data)}")
            
            if args.platform == 'twitter':
                with_media = sum(1 for item in scraped_data if item.get('has_media', False))
                total_media = sum(item.get('media_count', 0) for item in scraped_data)
                
                # Count by modality
                modality_counts = {}
                for item in scraped_data:
                    if item.get('has_media') and item.get('multimedia', {}).get('items'):
                        for media_item in item['multimedia']['items']:
                            modality = media_item.get('modality', 'unknown')
                            modality_counts[modality] = modality_counts.get(modality, 0) + 1
                
                print(f"   • Items with media: {with_media}")
                print(f"   • Total media items: {total_media}")
                if modality_counts:
                    modality_summary = ', '.join([f"{count} {modality}" for modality, count in modality_counts.items()])
                    print(f"   • Media by modality: {modality_summary}")
            
            if image_results:
                processed_images = len(image_results)
                successful_processing = sum(1 for r in image_results if r.get('processing_completed', False))
                print(f"   • Images processed: {processed_images}")
                print(f"   • Successful processing: {successful_processing}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Scraping interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 