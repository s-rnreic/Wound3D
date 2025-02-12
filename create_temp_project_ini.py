import configparser
import os

def setup_project_ini(workspace_path: str, *keywords) -> str:
    ini_file_command = f'call colmap project_generator --output_path "{workspace_path}\\project.ini" --quality high '
    os.system(ini_file_command)
    
    db_file_command = f'call colmap database_creator --database_path "{workspace_path}\\database.db"'
    os.system(db_file_command)
    
    project_ini_path = os.path.join(workspace_path, 'project.ini')
    database_path = os.path.join(workspace_path, 'database.db')
    
    with open(project_ini_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    if not lines[0].startswith('['):
        lines.insert(0, '[HEADER]\n')
    
    with open(project_ini_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    config = configparser.ConfigParser()
    config.read(project_ini_path)
    
    if 'HEADER' not in config:
        config['HEADER'] = {}
    
    config['HEADER']['database_path'] = database_path
    
    with open(project_ini_path, 'w', encoding='utf-8') as configfile:
        config.write(configfile)


    return project_ini_path
   
def insert_lines_at_beginning_of_ini_file(file_path: str, lines: list):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_lines = file.readlines()
    
    file_lines = lines + file_lines
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(file_lines)
    
    return file_path

def insert_keywords_to_project_ini(ini_path: str, keyword: str, parameter: str, value: str):
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    if keyword not in config:
        raise ValueError(f"Keyword [{keyword}] not found in the ini file.")
    
    config[keyword][parameter] = value
    
    with open(ini_path, 'w', encoding='utf-8') as configfile:
        config.write(configfile)
    
    return ini_path

def delete_keywords_from_project_ini(ini_path: str, keyword: str, parameter: str):
    with open(ini_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    if not lines[0].startswith('['):
        lines.insert(0, '[HEADER]\n')
    
    with open(ini_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    if keyword not in config:
        raise ValueError(f"Keyword [{keyword}] not found in the ini file.")
    
    if parameter not in config[keyword]:
        raise ValueError(f"Parameter [{parameter}] not found in the keyword [{keyword}] in the ini file.")
    
    del config[keyword][parameter]
    
    with open(ini_path, 'w', encoding='utf-8') as configfile:
        config.write(configfile)
    
    with open(ini_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    lines = [line for line in lines if '[HEADER]' not in line]
    
    with open(ini_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    return ini_path

def create_temp_project_ini(ini_path:str, *keywords):
    with open(ini_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    if not lines[0].startswith('['):
        lines.insert(0, '[HEADER]\n')
    
    temp_ini_path = os.path.join(os.path.dirname(ini_path), 'TempProject.ini')
    with open(temp_ini_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    config = configparser.ConfigParser()
    config.read(temp_ini_path)
    
    temp_config = configparser.ConfigParser()
    
    if 'HEADER' in keywords:
        if 'HEADER' in config:
            temp_config['HEADER'] = config['HEADER']
        keywords = [kw for kw in keywords if kw != 'HEADER']
    
    for keyword in keywords:
        if keyword not in config:
            raise ValueError(f"Keyword [{keyword}] not found in the ini file.")
        temp_config[keyword] = config[keyword]
    
    with open(temp_ini_path, 'w', encoding='utf-8') as temp_file:
        temp_config.write(temp_file)
    
    # Remove [HEADER] from the final ini file
    with open(temp_ini_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    lines = [line for line in lines if '[HEADER]' not in line]
    
    with open(temp_ini_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print(f"TempProject.ini created at {temp_ini_path}")
    return temp_ini_path

def execute_tepm_project_ini(ini_path, *keywords):
    command = f'call colmap {keywords[0]} --project_path "{ini_path}"'
    os.system(command)

def standard_ini_pipline(workspace_path, image_path, mask_path, *keywords):
    ini_path=setup_project_ini(workspace_path)
    insert_keywords_to_project_ini(ini_path, 'HEADER', 'image_path', image_path)
    insert_keywords_to_project_ini(ini_path, 'ImageReader', 'mask_path', mask_path)
    insert_keywords_to_project_ini(ini_path, 'StereoFusion', 'mask_path', mask_path)
    
    return ini_path
    
    

if __name__ == "__main__":
    workspace_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test'
    image_path=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\images"
    mask_path=r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\mask"
    ini_path = r'C:\Users\Reich\LocalCoding\COLMAP_Pipline\test\project.ini'
    standard_ini_pipline(workspace_path=workspace_path, image_path=image_path, mask_path=mask_path)
    keywords = ['HEADER','ImageReader','SiftExtraction']
    # setup_project_ini(workspace_path)
    # insert_keywords_to_project_ini(ini_path, 'HEADER', 'image_path', image_path)
    temp_ini_path =create_temp_project_ini(ini_path, *keywords)
    
    # execute_tepm_project_ini(temp_ini_path, 'feature_extractor')
