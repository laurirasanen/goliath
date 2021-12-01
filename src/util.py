import glob
import os


def get_latest_path():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(cur_dir, "../models")
    files = glob.glob(f"{models_dir}/model_*.zip")
    if len(files) > 0:
        # Assume sorted (datetime in name)
        print(f"Latest model: {files[-1]}")
        return files[-1]
    return None