import logging
import os
import re
import time
from urllib.parse import urljoin

import click
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_transcript(url):
    logger.info(f"Fetching transcript from URL: {url}")
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract year and number from the URL
    try:
        parts = url.split("/play/")
        year = parts[1].split("/")[0]
        number = parts[1].split("/")[1]
    except IndexError:
        logger.error(f"Failed to extract year or number from URL: {url}")
        return None

    # Extract talk title from OpenGraph metadata
    og_title_tag = soup.find("meta", property="og:title")
    talk_title = og_title_tag["content"] if og_title_tag else "Unknown Title"

    transcript_section = soup.find("section", id="transcript-content")
    if not transcript_section:
        logger.warning(f"No transcript section found for URL: {url}")
        return None

    # Group sentences by paragraph (<p>)
    paragraphs = []
    for p in transcript_section.find_all("p"):
        paragraphs.append(p.get_text(strip=True))

    logger.info(f"Transcript successfully extracted for URL: {url}")

    # Return the transcript object
    return {
        "year": year,
        "number": number,
        "talk_title": talk_title,
        "transcript": "\n".join(paragraphs),
        "url": url,  # Include the URL in the transcript object
    }


def extract_code_samples(soup):
    code_section = soup.find("li", class_="supplement sample-code")
    if not code_section:
        return []

    samples = []
    for container in code_section.find_all("li", class_="sample-code-main-container"):
        info_tag = container.find("p")
        timestamp = ""
        title = "Untitled"
        if info_tag:
            text = info_tag.get_text(" ", strip=True)
            match = re.match(r"^(\d{1,2}:\d{2}) - (.+)", text)
            if match:
                timestamp = match.group(1)
                title = match.group(2)
        code = container.find("pre", class_="code-source")
        code_text = code.get_text() if code else ""
        if code_text:
            samples.append({"timestamp": timestamp, "title": title, "code": code_text})
    return samples


def get_video_links(base_url):
    logger.info(f"Fetching video links from base URL: {base_url}")
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    video_links = []

    # Find all anchor tags with the expected class
    for a in soup.find_all(
        "a",
        class_="vc-card tile tile-rounded grid-item large-span-4 medium-span-6 small-span-12",
    ):
        href = a.get("href")
        if href:
            full_url = urljoin(base_url, href)
            video_links.append(full_url)

    logger.info(f"Found {len(video_links)} video links at base URL: {base_url}")
    return video_links


def save_transcript(transcript_object):
    # Ensure the transcripts directory exists
    os.makedirs("./transcripts", exist_ok=True)

    # Sanitize the title for filename
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", transcript_object["talk_title"]).strip(
        "_"
    )
    filename = (
        f"{transcript_object['year']}-{transcript_object['number']}-{safe_title}.md"
    )
    filepath = os.path.join("./transcripts", filename)

    content = (
        f"YEAR: {transcript_object['year']}\n"
        f"TITLE: {transcript_object['talk_title']}\n"
        f"URL: {transcript_object['url']}\n"  # Include the URL in the saved transcript header
        f"CONTENT:\n\n"
        f"{transcript_object['transcript']}\n"
    )

    if "code_samples" in transcript_object:
        content += "\n\nCODE SAMPLES:\n"
        for idx, sample in enumerate(transcript_object["code_samples"], 1):
            content += (
                f"\n--- Code Sample {idx} ---\n"
                f"**Time**: {sample['timestamp']}\n"
                f"**Title**: {sample['title']}\n\n"
                f"```swift\n{sample['code']}\n```\n"
            )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Transcript saved to {filepath}")


@click.command()
@click.option("--base_url", required=False, help="Base URL to fetch video links.")
@click.option("--url", required=False, help="Single video page URL to extract.")
def main(base_url, url):
    """Main function to extract transcripts from a base URL or a single video URL."""
    if url and base_url:
        logger.error("Please specify only one of --url or --base_url.")
        return
    elif url:
        transcript_object = extract_transcript(url)
        if transcript_object:
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            transcript_object["code_samples"] = extract_code_samples(soup)
            save_transcript(transcript_object)
        return
    elif base_url:
        logger.info(f"Starting transcript extraction process for base URL: {base_url}")
        links = get_video_links(base_url)
        for link in links:
            transcript_object = extract_transcript(link)
            if transcript_object:
                soup = BeautifulSoup(requests.get(link).text, "html.parser")
                transcript_object["code_samples"] = extract_code_samples(soup)
                save_transcript(transcript_object)
            else:
                logger.warning("Transcript extraction failed for one or more links")
            time.sleep(1)
    else:
        logger.error("Please specify either --url or --base_url.")


if __name__ == "__main__":
    main()
    # base_url = "https://developer.apple.com/videos/wwdc2025/"
    # logger.info("Starting transcript extraction process")
    # links = get_video_links(base_url)
    # for link in links:
    #     transcript_object = extract_transcript(link)
    #     if transcript_object:
    #         logger.info(
    #             f"Transcript extraction completed successfully: {transcript_object['talk_title']}"
    #         )
    #         save_transcript(transcript_object)
    #     else:
    #         logger.warning("Transcript extraction failed for one or more links")
    #     time.sleep(1)  # Add a 1-second delay between requests
