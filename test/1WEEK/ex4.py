import sys

for i in range(int(sys.argv[1])):

    fr = open(sys.argv[1], "r")
    fw = open("dst.txt", "w")

#    for line in fr:
#        fw.write(line)

fr.close()
fw.close()