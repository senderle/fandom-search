import os

for x in os.listdir('results/') :
    if not x.startswith('.'):
        os.system('`python format_helper.py' + ' ' + x.replace('-', ' ', 1) + '`')
