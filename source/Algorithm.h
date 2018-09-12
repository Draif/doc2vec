#pragma once
#include "NeuralNetwork.h"
#include "Vocabulary.h"
#include "Doc2Vec.h"

#include <vector>
#include <cassert>
#include <cmath>
#include <memory>

double VectorSimilarity(const TLayerVector<double>& vec1, const TLayerVector<double>& vec2);
double VectorDistance(const TLayerVector<double>& vec1, const TLayerVector<double>& vec2);

struct TSimilarObject {
    TSimilarObject(double similarity, unsigned int index)
        : Similarity(similarity)
        , Index(index)
    {}

    bool operator<(const TSimilarObject& another) const {
        return Similarity < another.Similarity;
    }

    double Similarity;
    unsigned int Index;
};

struct TSimilarWordObject : public TSimilarObject {
    TSimilarWordObject(const TWord& word, const TSimilarObject& simObject)
        : TSimilarObject(simObject)
        , Word(word)
    {}

    TWord Word;
};

struct TSimilarDocumentObject: public TSimilarObject {
    TSimilarDocumentObject(const std::shared_ptr<TDocument>& doc, const TSimilarObject& simObject)
        : TSimilarObject(simObject)
        , Document(doc)
    {}

    std::shared_ptr<TDocument> Document;
};

std::vector<TSimilarObject> FindSimilarObjects(const std::vector<double>& targetVec, const TLayer<double>& layer, unsigned int num);

std::vector<TSimilarWordObject> FindSimilarWords(const TDoc2Vec& doc2VecModel, const std::string& word, unsigned int num);
std::vector<TSimilarDocumentObject> FindSimilarDocs(const TDoc2Vec& doc2VecModel, unsigned int docIndex, unsigned int num);

void FindAndPrintSimilarWords(const TDoc2Vec& doc2VecModel, const std::string& word, unsigned int num);
void FindAndPrintSimilarDocs(const TDoc2Vec& doc2VecModel, unsigned int docIndex, unsigned int num);
void FindAndPrintSimilarDocs(const TDoc2Vec& doc2VecModel, const std::string& docTag, unsigned int num);

void PrintWordVector(const TDoc2Vec& doc2VecModel, const std::string& word);
void PrintDocVector(const TDoc2Vec& doc2VecModel, const std::string& docTag);
