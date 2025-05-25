import os

def copy_source_code(src_dir, dst_dir):
    """
    Copy source code files from src_dir to dst_dir.
    """
    dst_dir = os.path.join(dst_dir, 'source')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file in ['train.py', 'train2.py', 'test.py']:
        src_file = os.path.join('.', file)
        dst_file = os.path.join(dst_dir, file)
        with open(src_file, 'r') as fsrc:
            with open(dst_file, 'w') as fdst:
                fdst.write(fsrc.read())
    # Copy the entire src directory
    src_dir = os.path.join('.', 'src')
    dst_dir = os.path.join(dst_dir, 'src')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file in os.listdir(src_dir):
        if file.endswith('.py'):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            with open(src_file, 'r') as fsrc:
                with open(dst_file, 'w') as fdst:
                    fdst.write(fsrc.read())
