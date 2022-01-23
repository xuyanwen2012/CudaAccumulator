all:
	nvcc -c -O3 accumulator.cu
	nvcc -o a.out main.cpp accumulator.o

clean:
	rm *.o *.out
