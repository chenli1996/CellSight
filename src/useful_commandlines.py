I want to kill all python process:
44785 ttys022    0:00.12 /bin/zsh -il
52171 ttys022    0:05.90 python test_cluester.py
48092 ttys026    0:57.24 python test_voxel_grid.py
52248 ttys026    0:00.19 /bin/zsh -il
65289 ttys026    0:28.12 python test_cluester.py
28844 ttys028    0:00.04 -zsh
70566 ttys031    0:00.16 /bin/zsh -il

write a CML to kill all python process:
kill -9 $(ps | grep '[p]ython' | awk '{print $1}')