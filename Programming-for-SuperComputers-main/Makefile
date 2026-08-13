BIN               := Stream

# compilers
NVCC              := nvcc
NVCCFLAGS	  += -O3 -arch=sm_80

Stream: Stream.cu
	$(NVCC) $(NVCCFLAGS) $<  $(INCLUDES)   -o  $@

clean:
	rm -f $(BIN)
