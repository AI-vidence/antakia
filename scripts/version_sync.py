version = "*"
file_name = 'tst_pyproject.toml'
with open(file_name, 'r') as pyproject:
    new_file = []
    for line in pyproject.readlines():
        if 'version' in line:
            version = line.split('"')[1]
        if line[:12] == 'antakia-core':
            new_file.append(f'antakia-core = "{version}"\n')
        else:
            new_file.append(line)
with open(file_name, 'w') as pyproject:
    pyproject.write(''.join(new_file))
