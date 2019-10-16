import os

for x in os.listdir('results/') :
    if not x.startswith('.'):
        os.system('`python vis_helper.py' + ' ' + x.replace('-', ' ', 1) + '`')
