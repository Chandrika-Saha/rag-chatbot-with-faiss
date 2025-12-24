import requests
import json
import re


def crawl_meditations(url, output_file='meditations.json'):
    """
    Crawl and parse Marcus Aurelius' Meditations from MIT Classics

    Args:
        url: URL to the text file
        output_file: Output JSON filename
    """
    try:
        # Fetch the content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text = response.text

        # Split into lines and clean
        lines = [line.strip() for line in text.split('\n')]

        # Structure to hold the data
        data = {
            'title': 'The Meditations of Marcus Aurelius',
            'source': url,
            'books': []
        }

        current_book = None
        current_paragraph = []

        # Patterns to skip
        skip_patterns = [
            r'^[-=_*]+$',  # Lines with only separators
            r'^[\s-]+$',  # Lines with only dashes and spaces
            r'Translated by George Long',
            r'Copyright',
            r'^\d+\.$',  # Just numbers like "1."
        ]

        for line in lines:
            # Skip lines matching skip patterns
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue

            # Detect book headers - strict format: "BOOK ONE", "BOOK TWO", etc.
            if re.match(r'^BOOK\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE)$', line,
                        re.IGNORECASE):
                # Save previous paragraph if exists
                if current_paragraph and current_book:
                    para_text = ' '.join(current_paragraph)
                    if len(para_text) > 20:  # Only add substantial paragraphs
                        current_book['paragraphs'].append(para_text)
                    current_paragraph = []

                # Start new book
                current_book = {
                    'book_title': line,
                    'paragraphs': []
                }
                data['books'].append(current_book)

            elif line and current_book:  # Non-empty line
                current_paragraph.append(line)

            elif not line and current_paragraph and current_book:  # Empty line = end of paragraph
                para_text = ' '.join(current_paragraph)
                if len(para_text) > 20:  # Only add substantial paragraphs
                    current_book['paragraphs'].append(para_text)
                current_paragraph = []

        # Add last paragraph if exists
        if current_paragraph and current_book:
            para_text = ' '.join(current_paragraph)
            if len(para_text) > 20:
                current_book['paragraphs'].append(para_text)

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Successfully saved to {output_file}")
        print(f"✓ Total books: {len(data['books'])}")
        print(f"✓ Total paragraphs: {sum(len(book['paragraphs']) for book in data['books'])}")

        # Preview first book
        if data['books']:
            print(f"\nFirst book: {data['books'][0]['book_title']}")
            print(f"Paragraphs in first book: {len(data['books'][0]['paragraphs'])}")
            if data['books'][0]['paragraphs']:
                print(f"First paragraph preview: {data['books'][0]['paragraphs'][0][:150]}...")

        return data

    except requests.RequestException as e:
        print(f"✗ Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"✗ Error processing data: {e}")
        return None


# Run the crawler
if __name__ == '__main__':
    url = 'https://classics.mit.edu/Antoninus/meditations.mb.txt'
    data = crawl_meditations(url)