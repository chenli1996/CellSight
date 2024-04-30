import cProfile
import pstats
from node_feature import *

def main():
    # your code here
    for i in range(1000000):
        a = 1
    return

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    # main()
    generate_node_feature()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)

# To run the profiler, run the following command:
# python -m cProfile -o profile_output.prof your_script.py

# 
# import pstats
# p = pstats.Stats('../result/profile_output.prof')
# p.sort_stats('cumulative').print_stats(30)

# or visualize the profile
# snakeviz profile_output.prof


# or real time profiling
# py-spy top -- python your_script.py

# kill -9 $(ps aux | grep '[p]ython' | awk '{print $2}')
# kill -9 $(ps | grep '[p]ython' | awk '{print $1}')
