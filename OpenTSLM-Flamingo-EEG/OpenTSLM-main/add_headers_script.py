import os

EXCLUDE_DIRS = {'data', 'venv', '.git', '__pycache__', 'results', '.gemini'}
EXTENSIONS = {'.py', '.sh', '.md', '.txt'}

def should_process(path):
    parts = path.split(os.sep)
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
    _, ext = os.path.splitext(path)
    return ext in EXTENSIONS

def get_comment(path, ext):
    if ext == '.md':
        return f"<!-- File: {path} -->\n"
    else:
        return f"# File: {path}\n"

def main():
    root_dir = "."
    count = 0
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            path = os.path.join(root, file)
            # Normalize path to rely on relative path from current dir
            rel_path = os.path.relpath(path, root_dir)
            
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
                
            if should_process(rel_path):
                _, ext = os.path.splitext(rel_path)
                header = get_comment(rel_path, ext)
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Avoid double adding
                    if content.startswith(header.strip()):
                         continue
                         
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(header + content)
                    print(f"Added header to {rel_path}")
                    count += 1
                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")

    print(f"Processed {count} files.")

if __name__ == "__main__":
    main()
