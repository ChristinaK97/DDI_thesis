import requests

repo = 'https://raw.githubusercontent.com/ChristinaK97/DDI_thesis_files/main/'

def download_files(files):
    if isinstance(files, str):
        files = [files]
    for file in files:
        make_request(file)

def make_request(file):
    file_repo = repo + split_path(file)
    print("Downloading", file_repo)
    responce = requests.get(file_repo)
    with open(file, 'wb') as writer:
        writer.write(responce.content)

def split_path(file_path):
    index = file_path.find('data')
    return file_path[index:]















