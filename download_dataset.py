import urllib.request
import zipfile
import shutil
import os
print('Downloading ShapeNET...')

url = 'http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip'
urllib.request.urlretrieve(url, '3DShapeNetsCode.zip')

print('Extracting to 3DShapeNets...')
with zipfile.ZipFile("3DShapeNetsCode.zip","r") as zip_ref:
    zip_ref.extractall(".")

os.rename('3DShapeNets\\volumetric_data', 'train')