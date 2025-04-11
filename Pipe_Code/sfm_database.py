import os
import sqlite3
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# Creates and manages the database of the colmap/glomap Pipeline
# TODO: add reference tables from the database.db (cameras,image,features,pose_prior etc.) --> are they nessessary?
def setup_database(database_path: str) -> str:
    
    if database_path.endswith('.db'):
        pass
    elif os.path.isdir(database_path) or database_path.endswith('/') or database_path.endswith('\\'):
        database_path = os.path.join(database_path, 'Trial.db')
    # Connect to the database (it will be created if it doesn't exist)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Create the Trial table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Trial (
            id INTEGER PRIMARY KEY,
            experiment_description TEXT,
            folder_path TEXT,
            project_ini TEXT,
            cameras TEXT,
            image TEXT,
            Points3D TEXT,
            resize_factor REAL,
            fraction_of_images REAL,
            Observations INTEGER,
            mean_repo_error REAL,
            mean_track_length REAL,
            mean_observations_per_image REAL,
            elapsed_time REAL,
            Trial_date TEXT DEFAULT (DATETIME('now', 'localtime')),
            MOS INTEGER CHECK (MOS >= 1 AND MOS <= 10),
            Wound_MOS INTEGER CHECK (Wound_MOS >= 1 AND Wound_MOS <= 10),
            Density_MOS INTEGER CHECK (Density_MOS >= 1 AND Density_MOS <= 10),
            Homogeneity_MOS INTEGER CHECK (Homogeneity_MOS >= 1 AND Homogeneity_MOS <= 10),
            COLOR_MOS INTEGER CHECK (COLOR_MOS >= 1 AND COLOR_MOS <= 10),
            Noise_MOS INTEGER CHECK (Noise_MOS >= 1 AND Noise_MOS <= 10),
            branch TEXT,
            commit_hash TEXT
        )
    ''')
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()
        
    return database_path

def load_existing_experiments(database_path: str, folder_path: str, experiment_description: str, **kwargs) -> None:
    
    if not database_path.endswith('.db') or not any(fname.endswith('.db') for fname in os.listdir(os.path.dirname(database_path))):
        database_path = setup_database(database_path)
        
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Load project.ini
    project_ini_path = os.path.join(folder_path, 'total_project.ini')
    with open(project_ini_path, 'r') as file:
        project_ini = file.read()
    
    # Load binary files
    sparse_folder = os.path.join(folder_path, 'sparse', '0')
    cameras_path = os.path.join(sparse_folder, 'cameras.bin')
    images_path = os.path.join(sparse_folder, 'images.bin')
    points3D_path = os.path.join(sparse_folder, 'points3D.bin')
    
    result = subprocess.run(f'colmap model_analyzer --path "{sparse_folder}"', shell=True, capture_output=True, text=True)
    print(result)
     # Analyze the output
    output_lines = result.stderr.split('\n')
    observations = None
    mean_track_length = None
    mean_observations_per_image = None
    mean_reprojection_error = None
    
    for line in output_lines:
        if 'Observations' in line:
            observations = int(line.split(':')[-1].strip())
        elif 'Mean track length' in line:
            mean_track_length = float(line.split(':')[-1].strip())
        elif 'Mean observations per image' in line:
            mean_observations_per_image = float(line.split(':')[-1].strip())
        elif 'Mean reprojection error' in line:
            mean_reprojection_error = float(line.split(':')[-1].strip().replace('px', ''))

    for key, value in kwargs.items():
        if key == 'resize_factor':
            resize_factor = value
        elif key == 'fraction_of_images':
            fraction_of_images = value
        elif key == 'elapsed_time':
            elapsed_time = value

    branch = get_git_branch()
    commit_hash = get_git_commit()
    mos = None  # Assuming MOS is not provided in kwargs

    # Insert data into the database
    cursor.execute('''
        INSERT INTO Trial (experiment_description, project_ini, cameras, image, Points3D, Observations, mean_repo_error, mean_track_length, mean_observations_per_image, resize_factor, fraction_of_images, elapsed_time, MOS, branch, commit_hash, folder_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (experiment_description, project_ini, cameras_path, images_path, points3D_path, observations, mean_reprojection_error, mean_track_length, mean_observations_per_image, resize_factor, fraction_of_images, elapsed_time, mos, branch, commit_hash, folder_path))
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Get current Git branch
def get_git_branch():
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        return None  # Not in a Git repository

# Get latest commit hash
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        return None  # Not in a Git repository
    
def insert_MOS(database_path: str, mos_values: dict, trial_id: int) -> None:
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    for mos_type, score in mos_values.items():
        cursor.execute(f'''
            UPDATE Trial
            SET {mos_type} = ?
            WHERE id = ?
        ''', (score, trial_id))
    
    conn.commit()
    conn.close()

def open_mos_window(database_path: str, trial_id: int =None):
    def submit():
        try:
            trial_id = int(trial_id_entry.get())
            mos_values = {
                'MOS': int(mos_entry.get()),
                'Wound_MOS': int(wound_mos_entry.get()),
                'Density_MOS': int(density_mos_entry.get()),
                'Homogeneity_MOS': int(homogeneity_mos_entry.get()),
                'COLOR_MOS': int(color_mos_entry.get()),
                'Noise_MOS': int(noise_mos_entry.get())
            }
            for value in mos_values.values():
                if value < 1 or value > 10:
                    raise ValueError("MOS values must be between 1 and 10")
            insert_MOS(database_path, mos_values, trial_id)
            messagebox.showinfo("Success", "MOS values updated successfully")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    window = tk.Tk()
    window.title("Insert MOS Values")

    
        
    
    tk.Label(window, text="Trial ID:").grid(row=0, column=0)
    trial_id_entry = tk.Entry(window)
    # if not trial_id is None:
    #     trial_id_entry=trial_id
    trial_id_entry.grid(row=0, column=1)

    tk.Label(window, text="MOS:").grid(row=1, column=0)
    mos_entry = tk.Entry(window)
    mos_entry.grid(row=1, column=1)

    tk.Label(window, text="Wound MOS:").grid(row=2, column=0)
    wound_mos_entry = tk.Entry(window)
    wound_mos_entry.grid(row=2, column=1)

    tk.Label(window, text="Density MOS:").grid(row=3, column=0)
    density_mos_entry = tk.Entry(window)
    density_mos_entry.grid(row=3, column=1)

    tk.Label(window, text="Homogeneity MOS:").grid(row=4, column=0)
    homogeneity_mos_entry = tk.Entry(window)
    homogeneity_mos_entry.grid(row=4, column=1)

    tk.Label(window, text="COLOR MOS:").grid(row=5, column=0)
    color_mos_entry = tk.Entry(window)
    color_mos_entry.grid(row=5, column=1)
    
    tk.Label(window, text="Noise MOS:").grid(row=6, column=0)
    noise_mos_entry = tk.Entry(window)
    noise_mos_entry.grid(row=6, column=1)

    submit_button = tk.Button(window, text="Submit", command=submit)
    submit_button.grid(row=6, column=0, columnspan=2)

    window.mainloop()


if __name__ == '__main__':
    folder_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\1_reso_005_sharp_no_mask'
    database_path=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\Experiments\Trial.db"
    #setup_database(r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\benchmark')
    #load_existing_experiments(database_path, folder_path, experiment_description='Trails with 50% resolution and 50% of the number of images', resize_factor=0.5, fraction_of_images=0.5, elapsed_time=75)
    #insert_MOS(database_path, 6, 1)
    open_mos_window(database_path)