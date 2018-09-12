#include "NeuralNetwork.h"
#include "Common.h"

#include <fstream>
#include <string>

using namespace std;

template <typename T>
string TLayerVector<T>::CLASS_TAG = "TLayerVector";

template <typename T>
void TLayerVector<T>::Save(ofstream& out) const {
    out << CLASS_TAG << endl;
    out << Vector.size() << endl;
    for (const auto& it : Vector)
        out << it << SERIALIZE_DELIM;
    out << endl;
    out << CLASS_TAG << endl;
}

template <typename T>
void TLayerVector<T>::Load(ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TLayerVector::CLASS_TAG)
        throw runtime_error("TLayerVector::Load - wrong header.");
    unsigned int size;
    in >> size;
    getline(in, buf);

    Vector.resize(size);
    for (size_t i = 0; i < size; ++i)
        in >> Vector[i];

    getline(in, buf);
    getline(in, buf);
    if (buf != TLayerVector::CLASS_TAG)
        throw runtime_error("TLayerVector::Load - wrong tail.");
}

template <typename T>
string TLayer<T>::CLASS_TAG = "TLayer";

template <typename T>
void TLayer<T>::Save(std::ofstream& out) const {
    out << CLASS_TAG << endl;
    out << weights.size() << endl;
    for (const auto& vec : weights)
        vec.Save(out);
    out << CLASS_TAG << endl;

}

template <typename T>
void TLayer<T>::Load(std::ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TLayer::CLASS_TAG)
        throw runtime_error("TLayer::Load - wrong header.");
    unsigned int size;
    in >> size;
    getline(in, buf);

    weights.resize(size);
    for (size_t i = 0; i < size; ++i)
        weights[i].Load(in);

    getline(in, buf);
    if (buf != TLayer::CLASS_TAG)
        throw runtime_error("TLayer::Load - wrong tail.");
}

string TNeuralNetwork::CLASS_TAG = "TNeuralNetwork";

void TNeuralNetwork::Save(std::ofstream& out) const {
    out << TNeuralNetwork::CLASS_TAG << endl;
    out << MiddleDimension << SERIALIZE_DELIM << VocabularySize << SERIALIZE_DELIM << CorpusSize << endl;

    Syn0.Save(out);
    DSyn0.Save(out);
    Syn0Norm.Save(out);
    DSyn0Norm.Save(out);
    Syn1.Save(out);
    Syn1Neg.Save(out);

    out << TNeuralNetwork::CLASS_TAG << endl;
}

void TNeuralNetwork::Load(std::ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TNeuralNetwork::CLASS_TAG)
        throw runtime_error("TNeuralNetwork::Load - wrong header.");
    in >> MiddleDimension >> VocabularySize >> CorpusSize;
    getline(in, buf);

    Syn0.Load(in);
    DSyn0.Load(in);
    Syn0Norm.Load(in);
    DSyn0Norm.Load(in);
    Syn1.Load(in);
    Syn1Neg.Load(in);

    getline(in, buf);
    if (buf != TNeuralNetwork::CLASS_TAG)
        throw runtime_error("TNeuralNetwork::Load - wrong tail.");
}
