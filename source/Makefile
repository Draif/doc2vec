GCC=g++
CPPFLAGS= -std=c++11 -O4 -Wall -pthread
CPPFLAGS_DEBUG = -std=c++11 -g -O0 -Wall -pthread
TEST_OBJS = main.o Vocabulary.o Doc2Vec.o TrainThread.o Algorithm.o NeuralNetwork.o
SOURCE_FILES = main.cpp Vocabulary.cpp Doc2Vec.cpp TrainThread.cpp Algorithm.cpp NeuralNetwork.cpp

all: doc2vec

clean:
	rm -rf *.o $(TEST_OBJS) doc2vec test

test: $(TEST_OBJS)
	$(GCC) $(CPPFLAGS_DEBUG) $^ -o $@

doc2vec:
	$(GCC) $(CPPFLAGS) $(SOURCE_FILES) -o $@

%.o: %.cpp
	$(GCC) $(CPPFLAGS_DEBUG) -c $< -o $@
