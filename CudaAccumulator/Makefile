all:
	nvcc -c -O3 accumulator.cu
	nvcc -o a.out util.cpp bh_tree.cpp main.cpp accumulator.o

clean:
	rm *.o *.out
