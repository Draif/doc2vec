#pragma once
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <mutex>
#include <fstream>
#include <string>

template <typename T>
class TLayerVector {
public:
    TLayerVector() {}

    TLayerVector(unsigned int size, T value)
        : Vector(size, value)
    {}

    TLayerVector(const std::vector<T>& vec)
        : Vector(vec)
    {}

    TLayerVector(const TLayerVector& another)
        : Vector(another.Vector)
    {}

    void Lock() {
        Mutex.lock();
    }

    void Unlock() {
        Mutex.unlock();
    }

    T& operator[](unsigned int i) {
        return Vector[i];
    }

    const T& operator[](unsigned int i) const {
        return Vector[i];
    }

    unsigned int Size() const {
        return Vector.size();
    }

    typename std::vector<T>::const_iterator Begin() const {
        return Vector.cbegin();
    }

    typename std::vector<T>::const_iterator End() const {
        return Vector.cend();
    }

    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);
private:
    std::vector<T> Vector;
    std::mutex Mutex;
    static std::string CLASS_TAG;
};

template <class TObjClass>
class TSimpleLockGuard {
public:
    TSimpleLockGuard(TObjClass& obj)
        : Object(obj)
    {
        Object.Lock();
    }

    ~TSimpleLockGuard() {
        Object.Unlock();
    }

private:
    TObjClass& Object;
};

template <typename T>
class TLayerCreatorZeroPad {
public:
    std::vector<TLayerVector<T>> operator()(unsigned int layerSize, unsigned int dim) {
        return std::vector<TLayerVector<T>>(layerSize, TLayerVector<T>(dim, 0));
    }
};

template <class T>
class TLayerCreatorUniformRandom {
public:
    TLayerCreatorUniformRandom()
        : LowerBoarder(-0.5)
        , UpperBoarder(0.5)
    {}

    std::vector<TLayerVector<T>> operator()(unsigned int layerSize, unsigned int dim) {
        std::vector<TLayerVector<T>> layer;
        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(LowerBoarder, UpperBoarder);
        for (size_t i = 0; i < layerSize; ++i) {
            std::vector<T> tmpVector(dim);
            for (auto& it : tmpVector)
                it = distribution(generator);
            layer.emplace_back(tmpVector);
        }
        return layer;
    }
private:
    T LowerBoarder, UpperBoarder;
};


template <typename T>
class TLayer {
public:
    TLayer() {}

    template <class LayerCreator = TLayerCreatorZeroPad<T>>
    TLayer(unsigned int size, unsigned int dim, LayerCreator layerCreator = LayerCreator())
        : weights(layerCreator(size, dim))
    {}

    unsigned int Size() const {
        return weights.size();
    }

    TLayerVector<T>& operator[](unsigned int i) {
        if (i >= weights.size())
            throw std::runtime_error("Layer operator[] - out of range");
        return weights[i];
    }

    const TLayerVector<T>& operator[](unsigned int i) const {
        if (i >= weights.size())
            throw std::runtime_error("Layer operator[] - out of range");
        return weights[i];
    }

    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);
private:
    std::vector<TLayerVector<T>> weights;
    static std::string CLASS_TAG;
};

class TNeuralNetwork {
public:
    TNeuralNetwork() {}

    TNeuralNetwork(
        unsigned int vocabSize
        , unsigned int corpusSize
        , unsigned int dim
    )
        : MiddleDimension(dim)
        , VocabularySize(vocabSize)
        , CorpusSize(corpusSize)
        , Syn0(VocabularySize, MiddleDimension, TLayerCreatorUniformRandom<double>())
        , DSyn0(CorpusSize, MiddleDimension, TLayerCreatorUniformRandom<double>())
        , Syn0Norm(VocabularySize, MiddleDimension)
        , DSyn0Norm(CorpusSize, MiddleDimension)
        , Syn1(VocabularySize, MiddleDimension)
        , Syn1Neg(VocabularySize, MiddleDimension)
    {}

    void Normalize() {
        NormalizeLayer(Syn0, Syn0Norm);
        NormalizeLayer(DSyn0, DSyn0Norm);
    }

    TLayerVector<double>& GetDocumentVector(unsigned int docIndex) {
        if (docIndex >= DSyn0.Size())
            throw std::runtime_error("GetDocumentVector: out of range");
        return DSyn0[docIndex];
    }

    TLayerVector<double>& GetWordVector(unsigned int wordIndex) {
        if (wordIndex >= Syn0.Size())
            throw std::runtime_error("GetWordVector: out of range");
        return Syn0[wordIndex];
    }

    TLayerVector<double>& GetNegativeSampleVector(unsigned int index) {
        if (index >= Syn1Neg.Size())
            throw std::runtime_error("GetNegativeSampleVector: out of range");
        return Syn1Neg[index];
    }

    TLayerVector<double>& GetHierarchicalSoftmaxVector(unsigned int index) {
        if (index >= Syn1.Size())
            throw std::runtime_error("GetHierarchicalSoftmaxVector: out of range");
        return Syn1[index];
    }

    TLayerVector<double>& GetDocumentNormVector(unsigned int docIndex) {
        if (docIndex >= DSyn0Norm.Size())
            throw std::runtime_error("GetDocumentNormVector: out of range");
        return DSyn0Norm[docIndex];
    }

    const TLayerVector<double>& GetDocumentNormVector(unsigned int docIndex) const {
        if (docIndex >= DSyn0Norm.Size())
            throw std::runtime_error("GetDocumentNormVector: out of range");
        return DSyn0Norm[docIndex];
    }

    TLayerVector<double>& GetWordNormVector(unsigned int wordIndex) {
        if (wordIndex >= Syn0Norm.Size())
            throw std::runtime_error("GetWordNormVector: out of range");
        return Syn0Norm[wordIndex];
    }

    const TLayerVector<double>& GetWordNormVector(unsigned int wordIndex) const {
        if (wordIndex >= Syn0Norm.Size())
            throw std::runtime_error("GetWordNormVector: out of range");
        return Syn0Norm[wordIndex];
    }

    const TLayer<double>& GetWordsNormLayer() const {
        return Syn0Norm;
    }

    const TLayer<double>& GetDocsNormLayer() const {
        return DSyn0Norm;
    }

    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);

private:
    void NormalizeLayer(const TLayer<double>& layer, TLayer<double>& normLayer) {
        assert(layer.Size() == normLayer.Size());
        for (size_t i = 0; i < layer.Size(); ++i) {
            double len = 0;
            for (size_t j = 0; j < MiddleDimension; ++j)
                len += layer[i][j] * layer[i][j];
            len = sqrt(len);
            for (size_t j = 0; j < MiddleDimension; ++j)
                normLayer[i][j] = layer[i][j] / len;
        }
    }

private:
    unsigned int MiddleDimension, VocabularySize, CorpusSize;
    TLayer<double> Syn0, DSyn0;
    TLayer<double> Syn0Norm, DSyn0Norm;
    TLayer<double> Syn1, Syn1Neg;

    static std::string CLASS_TAG;
};
