version = "*"
file_name = 'pyproject.toml'
with open(file_name, 'r') as pyproject:
    new_file = []
    for line in pyproject.readlines():
        if 'version' in line:
            version = line.split('"')[1]
        if 'antakia-core' in line:
            if '{' in line.split('=')[1]:
                if line[:12] == 'antakia-core':
                    line = '#' + line
                new_file.append(line)
            else:
                new_file.append(f'antakia-core = "{version}"\n')
        elif 'antakia-ac' in line:
            if '{' in line.split('=')[1]:
                if line[:10] == 'antakia-ac':
                    line = '#' + line
                new_file.append(line)
            else:
                line = line[line.find('antakia-ac'):]
                new_file.append(line)
        else:
            new_file.append(line)
with open(file_name, 'w') as pyproject:
    pyproject.write(''.join(new_file))
