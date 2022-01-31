all:
	gpu

cpu: 
	gcc -std=c99 -c -O3 -o accumulator_c.o accumulator_c.c
	g++ -O3 -oc.out util.cpp bh_tree.cpp main.cpp accumulator_c.o

gpu:
	nvcc -c -O3 accumulator.cu
	gcc -c -O3 util.cpp -fopenmp
	nvcc -o a.out bh_tree.cpp main.cpp util.o accumulator.o

clean:
	rm *.o *.out
