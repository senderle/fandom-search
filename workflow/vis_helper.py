import sys
cmd = ('python ao3.py vis '
       '-o results/{0:}-{1:}/{2:}_reuse.html '
       'results/{0:}-{1:}/fandom-data-{1:}.csv')
cmd = cmd.format(sys.argv[1],
                 sys.argv[2],
                 sys.argv[2].replace('-', '_'))
print(cmd)

