import os
# get to right directory
os.chdir('/home/cnslab/imagenet/')

dir = os.listdir()
files = len(dir)
baseDir = "/home/cnslab/imagenet/"
print('Starting...')
for singFile in dir:
  if singFile.lower().endswith('.jpeg'):
    if not os.path.exists(baseDir + '/' + singFile[0:10]):
      os.makedirs(baseDir + '/' + singFile[0:10], 0o777)
      print('Made new folder')

    os.system("mv " + baseDir + singFile + " " + baseDir + singFile[0:10])

  files -= 1
  print(str(files) + " left")