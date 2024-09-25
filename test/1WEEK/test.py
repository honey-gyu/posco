import sys

#print(sys.argv[1]) # 오랜만이야 아그시 아그비

#for i in range(int(sys.argv[1])):
#    print("HEllO", i)
    
    
# new.txt를 그대로 복사해서 new22.txt를 생성
fr = open("new.txt", "r")
fw = open("new22.txt", "w")

for line in fr:
    fw.write(line)

fr.close()
fw.close()