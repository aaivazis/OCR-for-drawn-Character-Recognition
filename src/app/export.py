import json
import os
from datetime import datetime
from typing import List, Tuple


def save_strokes_to_json(num_strokes, strokes: List[List[Tuple[float, float, float]]], 
                        output_dir: str = "data/raw") -> str:
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data structure
    letter_data = {
        "timestamp": datetime.now().isoformat(),
        "num_strokes": num_strokes,
        "strokes": strokes  # List of strokes, each stroke is list of (x, y, t) tuples
    }
    
    # Generate filename with timestamp
    filename = f"letter_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(letter_data, f, indent=2)
    
    return filepath

