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
        ## Check the OS
        if os.name == 'nt':  # Windows
            # For Windows, ensure the command is properly quoted
            print("working on Windows")
            command = shlex.split(command)
            result = subprocess.run(command, check=True, shell=True)
            #print(result.stdout.decode())
        else:  # Assuming it's a Unix-like system (Linux/macOS)
            # For Unix-like systems, we should also be able to run the command directly
            print("working on Unix")
            command = shlex.split(command)
            result = subprocess.run(command, check=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing: {e}")
        
if __name__ == "__main__":
    # Example usage
    path=r"C://Users//Reich//LocalCoding//COLMAP_Pipline//Test//database.db"
    #command = f"colmap database_creator --database_path '{path}"
    command= "colmap -h"
    run_command(command)