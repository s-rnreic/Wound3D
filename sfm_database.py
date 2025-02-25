import os
import sqlite3
import subprocess

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
            project_ini TEXT,
            cameras BLOB,
            image BLOB,
            Points3D BLOB,
            resize_factor REAL,
            fraction_of_images REAL,
            Observations INTEGER,
            mean_repo_error REAL,
            mean_track_length REAL,
            mean_observations_per_image REAL,
            elapsed_time REAL,
            Trial_date TEXT DEFAULT (DATETIME('now', 'localtime')),
            MOS INTEGER CHECK (MOS >= 1 AND MOS <= 10),
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
    
    with open(cameras_path, 'rb') as file:
        cameras = file.read()
    with open(images_path, 'rb') as file:
        images = file.read()
    with open(points3D_path, 'rb') as file:
        points3D = file.read()
    
    result = subprocess.run(f"colmap model_analyzer --path {sparse_folder}", shell=True, capture_output=True, text=True)
    #print(result.stderr)
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
        INSERT INTO Trial (experiment_description, project_ini, cameras, image, Points3D, Observations, mean_repo_error, mean_track_length, mean_observations_per_image, resize_factor, fraction_of_images, elapsed_time, MOS, branch, commit_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (experiment_description, project_ini, cameras, images, points3D, observations, mean_reprojection_error, mean_track_length, mean_observations_per_image, resize_factor, fraction_of_images, elapsed_time, mos, branch, commit_hash))
    
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

def insert_MOS(database_path: str, score: int, id: int) -> None:
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE Trial
        SET MOS = ?
        WHERE id = ?
    ''', (score,id))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    folder_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\1_reso_005_sharp_no_mask'
    database_path=r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\Trial.db'
    #setup_database(r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\benchmark')
    #load_existing_experiments(database_path, folder_path, experiment_description='Trails with 50% resolution and 50% of the number of images', resize_factor=0.5, fraction_of_images=0.5, elapsed_time=75)
    insert_MOS(database_path, 6, 1)