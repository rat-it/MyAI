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

    with open(r'c:\AI-Projects\MyAI\nb_test_log.txt', 'w', encoding='utf-8') as log_file:
        for f in notebook_files:
            log_file.write(f"\n==========================================\n")
            log_file.write(f"Executing: {f}\n")
            print(f"Executing: {f}")
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    nb = nbformat.read(file, as_version=4)
                # execute
                ep.preprocess(nb, {'metadata': {'path': os.path.dirname(f)}})
                # write back
                with open(f, 'w', encoding='utf-8') as file:
                    nbformat.write(nb, file)
                log_file.write(f"Success: {f}\n")
            except Exception as e:
                log_file.write(f"Failed: {f}\n")
                log_file.write(traceback.format_exc() + "\n")
                failed_notebooks.append(f)
                
        log_file.write("\n\n================ SUMMARY ================\n")
        if not failed_notebooks:
            log_file.write("All notebooks executed successfully.\n")
        else:
            log_file.write(f"{len(failed_notebooks)} notebooks failed:\n")
            for nf in failed_notebooks:
                log_file.write(f" - {nf}\n")

if __name__ == '__main__':
    run_notebooks()
