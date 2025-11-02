# run-pipeline.py

import importlib
from rich.console import Console

# Import the files for each step
step0 = importlib.import_module("src.0-get-data")
step1 = importlib.import_module("src.1-clean-audio")
step2 = importlib.import_module("src.2-split-clips")
step3 = importlib.import_module("src.3-filter-and-balance")
step4 = importlib.import_module("src.4-extract-features")
step5 = importlib.import_module("src.5-train-model")

# setup console
console = Console()

def main():
    """ 
    Run the entire pipeline step by step.
    
    Args:
        none
    
    Returns:
        runs all steps in sequence
    
    Pipeline Steps:
        all 5 steps

    Expects:
        nothing, starts fresh recording session and saves data for next steps   
    """

    # Start with ASCII art title
    try:
        with open("images/ascii.txt", "r") as f:
            file_ascii_art = f.read()
            print(file_ascii_art)
    except FileNotFoundError:
        print("Error: 'images/ascii.txt' not found.")

    # console.rule("[bold red]Step 0/5: Get Data")
    # step0.main()
    
    console.rule("[bold red]Step 1/5: Clean Audio")
    step1.main()
    
    console.rule("[bold red]Step 2/5: Split Clips")
    step2.main()
    
    console.rule("[bold red]Step 3/5: Filter and Balance")
    step3.main()
    
    console.rule("[bold red]Step 4/5: Extract Features")
    step4.main()
    
    console.rule("[bold red]Step 5/5: Train Model")
    step5.main()
    
    console.rule("[bold green]Pipeline completed successfully!")
    

if __name__ == "__main__":
    main()