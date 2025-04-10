import os
import subprocess
import shlex

# Function to run command in a subprocess
def run_command(command):
    try:
        # Use shlex.split for safe command argument splitting
        command = shlex.split(command)
        # Run the command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing: {e}")