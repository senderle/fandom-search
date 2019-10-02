import sys
import os
import re

franchise = sys.argv[1]
movie = sys.argv[2]

if len(sys.argv) <= 3:
    movie_folder = 'results/{0:}-{1:}'.format(franchise, movie)
    data_folders = os.listdir(movie_folder)
    dates = sorted([f for f in data_folders if re.search(r'[0-9]{8}', f)])
    date = dates[-1]
else:
    date = sys.argv[3]

cmd = ('python ao3.py format '
       '-o results/{0:}-{1:}/fandom-data-{1:}.csv '
       'results/{0:}-{1:}/{2:}/match-6gram-{2:}.csv '
       'scripts/{0:}-{1:}.txt')
cmd = cmd.format(franchise,
                 movie,
                 date)
print(cmd)

