import os
import sys

#<obj> <dim> <fun> <inst>
if __name__ == '__main__':
    obj = int(sys.argv[1])
    dim = int(sys.argv[2])
    fun = int(sys.argv[3])-1
    inst = int(sys.argv[4])

    available_instances = [5641, 3854, 8370, 494, 1944, 9249, 2517, 2531, 5453, 2982, 670, 56, 6881, 1930, 5812]
    inst = available_instances[inst]

    if obj==2:
        fun += 5;

    print("./TopTrumpsExec " + str(inst) + " " + str(fun) + " 100")
    os.system("./TopTrumpsExec " + str(inst) + " " + str(fun) + " 100")


