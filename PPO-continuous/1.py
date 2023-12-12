"""
unzip google16k and rename it
"""
import glob
import os

current_dir=os.getcwd()
# print(current_dir)
# print(glob.glob(current_dir+'/*'))
# tgz_list=[f for f in glob.glob(current_dir+'/*') if f.endswith('.tgz')]
dir_list=[f for f in glob.glob(current_dir+'/assets/ycb/visual/*') ]
# tgz_list=[dir+'/google_16k' for dir in tgz_list if dir.endswith('py') is False]

for dir in dir_list:
                  print(dir)
                  print(glob.glob(dir+'/*.stl'))
                  file=glob.glob(dir+'/*.stl')[0]
                  os.system('mv '+file+' '+os.path.dirname(file)+'/textured_simple.mtl')
                  # file=glob.glob(dir+'/*.obj')[0]
                  # os.system('mv '+file+' '+os.path.dirname(os.path.dirname(file))+'/textured_simple.obj')
                  # file=glob.glob(dir+'/*.png')[0]
                  # os.system('mv '+file+' '+os.path.dirname(os.path.dirname(file))+'/textur_map.png')
                  # os.system(f'rm -r {dir}')
# print(len(tgz_list))
# for file in tgz_list:
#                   print(file)
#                   os.system('tar -xvzf '+file)