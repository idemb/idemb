BINFILES  = LearningMain

all: ${BINFILES}

run: all
	./LearningMain


-include .depend.mk




CC = $(CXX)
CXXFLAGS = -I..  -Wall -Wvla -g -pg -ggdb -std=c++11 -pthread 



#headers = \
	./SGDLearning.hpp \
	./Metrics.hpp \
	./Relations.hpp \
	./Distance.hpp \
	./NormalSGDLearning.hpp \
	./SparseSGDLearning.hpp \
	./CSVReader.hpp 
#	
	
	
	


LDLIBS = -larmadillo -lpthread


depend:
	-$(CXX) -M $(CXXFLAGS) *.cpp > .depend.mk


clean:
	-rm -f *.o *.a *.so  $(BINFILES) 
	
distclean: clean
	-rm -f .depend.mk