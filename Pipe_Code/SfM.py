from pathlib import Path
import sys
from typing import Optional
import pycolmap
sys.path.append(r"C:\Users\Reich\LocalCoding\COLMAP_Pipline\external")
from hloc.hloc.triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)
from hloc.hloc.utils.database import COLMAPDatabase

from hloc.hloc.reconstruction import (
    create_empty_db,
    get_image_ids,
    import_images,
    import_features,
    import_matches,
)


from hloc.hloc import extract_features, match_features, pairs_from_exhaustive

def LightGlue(image_path: Path, output_path: Path, feature_conf_name: Optional[str] ="superpoint_aachen", matcher_conf_name: Optional[str]= "superglue"):
    # Ensure output_path is a Path object
    output_path = Path(output_path)
    # Ensure the outputs directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    image_list = output_path / "image_list.txt"

    # Ensure the image list file exists and contains full paths to the images
    if not image_list.exists():
        with open(image_list, "w") as f:
            for image in Path(image_path).iterdir():  # Convert string to Path and use `iterdir()`
                if image.suffix in ['.jpg', '.png']:  # Check file extensions
                    f.write(image.name + "\n")  # Write only the file name with extension

    # Define a valid file path for the exhaustive pairs output
    exhaustive_pairs_file = Path(output_path / "pairs-exhaustive.txt")

    # Pass the file path instead of the directory to pairs_from_exhaustive.main
    pairs_from_exhaustive.main(output=exhaustive_pairs_file, image_list=image_list)

    feature_conf = extract_features.confs[feature_conf_name]
    matcher_conf = match_features.confs[matcher_conf_name]

    # Pass num_workers=0 to disable multiprocessing in DataLoader
    feature_path = extract_features.main(feature_conf, image_path, output_path)
    match_path = match_features.main(
        matcher_conf, exhaustive_pairs_file, feature_conf["output"], output_path
    )
    
    #delete the image_list file
    #image_list.unlink(missing_ok=True)
    #delete the pairs-exhaustive file
    #exhaustive_pairs_file.unlink(missing_ok=True)
    return feature_path, match_path, exhaustive_pairs_file
    
def Stream_into_db(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
):
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)
    import_images(image_dir, database, camera_mode)
    image_ids = get_image_ids(database)
    print(f"Image IDs: {image_ids}")
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)
