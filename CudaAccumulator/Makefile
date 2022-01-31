all:
	gpu

cpu: 
	gcc -std=c99 -c -O3 -o accumulator_c.o accumulator_c.c
	g++ -O3 -oc.out util.cpp bh_tree.cpp main.cpp accumulator_c.o -fopenmp

gpu:
	nvcc -c -O3 accumulator.cu
	nvcc -o a.out bh_tree.cpp util.cpp main.cpp accumulator.o -Xcompiler -fopenmp -O3

clean:
	rm *.o *.out
