import os
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import traceback

def run_notebooks():
    search_path = r'c:\AI-Projects\MyAI\**\*.ipynb'
    notebook_files = glob.glob(search_path, recursive=True)
    notebook_files = [f for f in notebook_files if '.ipynb_checkpoints' not in f]
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    failed_notebooks = []

    for f in notebook_files:
        print(f"\n==========================================")
        print(f"Executing: {f}")
        try:
            with open(f, 'r', encoding='utf-8') as file:
                nb = nbformat.read(file, as_version=4)
            # execute
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(f)}})
            # write back
            with open(f, 'w', encoding='utf-8') as file:
                nbformat.write(nb, file)
            print(f"Success: {f}")
        except Exception as e:
            print(f"Failed: {f}")
            print(traceback.format_exc())
            failed_notebooks.append(f)
            
    print("\n\n================ SUMMARY ================")
    if not failed_notebooks:
        print("All notebooks executed successfully.")
    else:
        print(f"{len(failed_notebooks)} notebooks failed:")
        for nf in failed_notebooks:
            print(f" - {nf}")

if __name__ == '__main__':
    run_notebooks()
