import os
import subprocess
import shlex
import shutil

# Function to run command in a subprocess
def run_command(command):
    # Check if 'colmap' is available in PATH
    if not shutil.which("colmap"):
        print("Error: 'colmap' executable not found. Please ensure it is installed and added to the system's PATH.")
        return
    try:
        # Use shlex.split for safe command argument splitting
        command = shlex.split(command)
        # Run the command
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing: {e}")
        
if __name__ == "__main__":
    # Example usage
    path=r"C://Users//Reich//LocalCoding//COLMAP_Pipline//Test//database.db"
    #command = f"colmap database_creator --database_path '{path}"
    command= "colmap -h"
    run_command(command)