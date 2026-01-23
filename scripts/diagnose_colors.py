"""
PDF Color Diagnostic Tool
Extracts all unique colors from PDFs to understand the actual color scheme.

Usage:
    python diagnose_colors.py --pdf path/to/pdf.pdf
    python diagnose_colors.py --pdf-dir path/to/pdfs  # Scan all PDFs
"""

import sys
from scripts.utils import map_color_to_cat
from pathlib import Path
import argparse
from collections import Counter

try:
    import fitz
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install PyMuPDF")
    sys.exit(1)


def rgb_to_hex(r, g, b):
    """Convert RGB to hex."""
    return f"#{r:02x}{g:02x}{b:02x}"


def analyze_pdf_colors(pdf_path, sample_phrases=None):
    """
    Extract all unique colors from a PDF and show sample text for each.
    
    Args:
        pdf_path: Path to PDF
        sample_phrases: Optional list of phrases to specifically track
    """
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).name
    
    # Track colors and their usage
    color_samples = {}  # {hex_color: [(text, page), ...]}
    phrase_colors = {}  # {phrase: (hex, rgb, text)}
    
    print(f"\n{'='*70}")
    print(f"📄 Analyzing: {pdf_name}")
    print(f"{'='*70}")
    
    for page_num in range(min(3, len(doc))):  # First 3 pages
        page = doc.load_page(page_num)
        text_dict = page.get_text('dict')
        
        for block in text_dict.get('blocks', []):
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue
                    
                    color = span.get('color')
                    if color is not None:
                        # Convert to RGB
                        r = (color >> 16) & 0xFF
                        g = (color >> 8) & 0xFF
                        b = color & 0xFF
                        hex_color = rgb_to_hex(r, g, b)
                        
                        # Store sample
                        if hex_color not in color_samples:
                            color_samples[hex_color] = []
                        
                        if len(color_samples[hex_color]) < 5:  # Keep 5 samples per color
                            color_samples[hex_color].append((text, page_num))
                        
                        # Track specific phrases
                        if sample_phrases:
                            for phrase in sample_phrases:
                                if phrase.lower() in text.lower():
                                    phrase_colors[phrase] = (hex_color, (r, g, b), text)
    
    # Print color report
    print(f"\n🎨 Found {len(color_samples)} unique colors:\n")
    
    # Sort by frequency (most common first)
    color_freq = {color: len(samples) for color, samples in color_samples.items()}
    sorted_colors = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)
    
    for hex_color, freq in sorted_colors:
        # Parse RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # Classify
        classification = classify_color(r, g, b)
        
        print(f"{hex_color}  RGB({r:3d}, {g:3d}, {b:3d})  [{classification:8s}]")
        
        # Show samples
        samples = color_samples[hex_color][:3]
        for text, page in samples:
            print(f"  └─ Page {page}: {text[:50]}")
        print()
    
    # Print phrase matches
    if phrase_colors:
        print(f"\n🔍 Phrase Color Matches:\n")
        for phrase, (hex_color, rgb, text) in phrase_colors.items():
            r, g, b = rgb
            classification = classify_color(r, g, b)
            print(f"'{phrase}'")
            print(f"  Color: {hex_color} RGB({r}, {g}, {b}) [{classification}]")
            print(f"  Text:  {text[:60]}")
            print()
    
    return color_samples, phrase_colors


def classify_color(r, g, b):
    def map_color_to_cat(rgb, debug=False):
        """
        Map RGB tuple to color category based on ACTUAL colors found in PDFs.
    
        Categories:
        - red: Negative credit factors (late payments, collections, high utilization)
        - green: Positive credit factors (good history, seasoned accounts)
        - black: Neutral text (headers, labels, standard info)
        - neutral: Ambiguous (scores, dates, other)
        """
        r, g, b = rgb
    
        # Exact match for known colors (most reliable)
        rgb_hex = '#{:02x}{:02x}{:02x}'.format(r, g, b).lower()
    
        # RED variants (negative factors)
        red_colors = {'#dc3545', '#dd5435', '#ad302e'}
        if rgb_hex in red_colors:
            if debug:
                print(f"  ✓ Exact red match: {rgb_hex}")
            return 'red'
    
        # GREEN variants (positive factors)
        green_colors = {'#428f4e', '#8cb955'}
        if rgb_hex in green_colors:
            if debug:
                print(f"  ✓ Exact green match: {rgb_hex}")
            return 'green'
    
        # BLACK/NEUTRAL base colors
        black_colors = {'#212529', '#374151'}
        if rgb_hex in black_colors:
            if debug:
                print(f"  ✓ Exact black match: {rgb_hex}")
            return 'black'
    
        # SCORE/NEUTRAL colors (orange, yellow)
        score_colors = {'#e8923e', '#f7cd47'}
        if rgb_hex in score_colors:
            if debug:
                print(f"  ✓ Score/neutral color: {rgb_hex}")
            return 'neutral'
    
        # Fallback: Threshold-based detection for unknown colors
        if debug:
            print(f"  ? Unknown color {rgb_hex}, using thresholds...")
    
        # RED: High red channel, low green/blue
        # Covers all three red variants: (220,53,69), (221,84,53), (173,48,46)
        if r > 150 and g < 100 and b < 80:
            if debug:
                print(f"    → Classified as RED (r={r}, g={g}, b={b})")
            return 'red'
    
        # GREEN: Green channel dominant and significantly higher than red/blue
        # Covers both green variants: (66,143,78), (140,185,85)
        if g > 70 and g > r + 30 and g > b + 30:
            if debug:
                print(f"    → Classified as GREEN (r={r}, g={g}, b={b})")
            return 'green'
    
        # BLACK: All channels very low
        if r < 60 and g < 70 and b < 90:
            if debug:
                print(f"    → Classified as BLACK (r={r}, g={g}, b={b})")
            return 'black'
    
        # Everything else is neutral (scores, dates, etc.)
        if debug:
            print(f"    → Classified as NEUTRAL (r={r}, g={g}, b={b})")
        return 'neutral'
    """Classify color for diagnostic purposes."""
    # Red
    if r > 150 and g < 100 and b < 100:
        return "RED"
    
    # Green (current logic)
    if g > 100 and g > r and g > b:
        return "GREEN"
    
    # Black
    if r < 50 and g < 50 and b < 50:
        return "BLACK"
    
    # Check if it's a green variant
    if g > r and g > b:
        return f"GREEN-ish"
    
    # Gray
    if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
        if r > 50:
            return "GRAY"
        else:
            return "BLACK"
    
    return "NEUTRAL"


def analyze_multiple_pdfs(pdf_dir, sample_phrases=None):
    """Analyze all PDFs in a directory."""
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    print(f"🚀 Analyzing {len(pdf_files)} PDFs for color usage")
    
    all_colors = Counter()
    
    for pdf_path in pdf_files:
        color_samples, phrase_colors = analyze_pdf_colors(pdf_path, sample_phrases)
        
        # Aggregate colors
        for hex_color in color_samples.keys():
            all_colors[hex_color] += 1
    
    # Print global color summary
    print(f"\n{'='*70}")
    print(f"📊 GLOBAL COLOR SUMMARY")
    print(f"{'='*70}\n")
    print(f"Colors used across all PDFs:\n")
    
    for hex_color, count in all_colors.most_common():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        classification = classify_color(r, g, b)
        
        print(f"{hex_color}  RGB({r:3d}, {g:3d}, {b:3d})  [{classification:8s}]  Used in {count}/{len(pdf_files)} PDFs")


def main():
    parser = argparse.ArgumentParser(description='Diagnose PDF colors')
    parser.add_argument('--pdf', type=str, help='Single PDF to analyze')
    parser.add_argument('--pdf-dir', type=str, help='Directory of PDFs to analyze')
    parser.add_argument('--phrases', type=str, nargs='+', help='Specific phrases to track')
    args = parser.parse_args()
    
    # Default phrases to track
    default_phrases = [
        'Past Due Not Late',
        'Ok Open Rev Depth',
        '1 Rev Late in 0-3 mo',
        'No Closed Rev Depth',
        'Credit Score',
        'Total Rev Usage > 55%',
        '6 Charged Off Accts'
    ]
    
    sample_phrases = args.phrases if args.phrases else default_phrases
    
    if args.pdf:
        analyze_pdf_colors(args.pdf, sample_phrases)
    elif args.pdf_dir:
        analyze_multiple_pdfs(args.pdf_dir, sample_phrases)
    else:
        print("ERROR: Must specify --pdf or --pdf-dir")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # Use the canonical map_color_to_cat for all color classification
    return map_color_to_cat((r, g, b)).upper()