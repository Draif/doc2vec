#include "Algorithm.h"

#include <vector>
#include <set>
#include <cassert>
#include <cmath>

using namespace std;

double VectorSimilarity(const TLayerVector<double>& vec1, const TLayerVector<double>& vec2) {
    assert(vec1.Size() == vec2.Size());
    double res = 0;
    for (size_t i = 0; i < vec1.Size(); ++i)
        res += vec1[i] * vec2[i];
    return res;
}


double VectorDistance(const TLayerVector<double>& vec1, const TLayerVector<double>& vec2) {
    assert(vec1.Size() == vec2.Size());
    double res = 0;
    for (size_t i = 0; i < vec1.Size(); ++i)
        res += pow(vec1[i] - vec2[i], 2);
    return sqrt(res);
}

vector<TSimilarObject> FindSimilarObjects(unsigned int targetIndex, const TLayer<double>& layer, unsigned int num) {
    set<TSimilarObject> heap;
    const auto& targetVec = layer[targetIndex];
    for (size_t i = 0; i < layer.Size(); ++i) {
        if (i == targetIndex)
            continue;

        double similarity = VectorSimilarity(targetVec, layer[i]);
        if (heap.size() < num) {
            heap.insert(TSimilarObject(similarity, i));
        } else {
            if (similarity > heap.begin()->Similarity) {
                heap.erase(heap.begin());
                heap.insert(TSimilarObject(similarity, i));
            }
        }
    }
    return vector<TSimilarObject>(heap.rbegin(), heap.rend());
}

vector<TSimilarWordObject> FindSimilarWords(const TDoc2Vec& doc2VecModel, const string& word, unsigned int num) {
    vector<TSimilarWordObject> res;
    TWord wordStruct;
    const auto& wordsVoc = doc2VecModel.GetWordsVocabulary();
    const auto& neuralNetwork = doc2VecModel.GetNeuralNetwork();
    auto normWord = NormalizeWord(word);

    if (!wordsVoc.GetWord(normWord, wordStruct))
        return res;

    const auto& layer = neuralNetwork.GetWordsNormLayer();
    auto similarObjects = FindSimilarObjects(wordStruct.Index, layer, num);

    for (const auto& similarObject : similarObjects) {
        if (!wordsVoc.GetWord(similarObject.Index, wordStruct))
            throw runtime_error("Cannot find object by index.");
        res.emplace_back(wordStruct, similarObject);
    }
    return res;
}

vector<TSimilarDocumentObject> FindSimilarDocs(const TDoc2Vec& doc2VecModel, unsigned int docIndex, unsigned int num) {
    vector<TSimilarDocumentObject> res;
    const auto& neuralNetwork = doc2VecModel.GetNeuralNetwork();
    const auto& docsHolder = doc2VecModel.GetDocsHolder();
    const auto& layer = neuralNetwork.GetDocsNormLayer();

    auto similarObjects = FindSimilarObjects(docIndex, layer, num);
    for (const auto& similarObject : similarObjects) {
        const auto& doc = docsHolder.GetDocument(similarObject.Index);
        res.emplace_back(doc, similarObject);
    }
    return res;
}

void FindAndPrintSimilarWords(const TDoc2Vec& doc2VecModel, const string& word, unsigned int num) {
    TWord wordStruct;
    const auto& wordsVoc = doc2VecModel.GetWordsVocabulary();
    auto normWord = NormalizeWord(word);

    if (!wordsVoc.GetWord(normWord, wordStruct)) {
        cout << "Word " << '"' << word << '"' << " isn't in vocabulary." << endl;
        return;
    }

    auto similarWords = FindSimilarWords(doc2VecModel, word, num);
    cout << '"' << word << '"' << " similar words:" << endl;
    if (similarWords.empty()) {
        cout << "No similar words were found" << endl;
        return;
    }
    for (const auto& simWord : similarWords)
        cout << "\t" << '"' << simWord.Word.Word << '"' << " -> " << simWord.Similarity << endl;
}

void FindAndPrintSimilarDocs(const TDoc2Vec& doc2VecModel, unsigned int docIndex, unsigned int num) {
    auto similarDocs = FindSimilarDocs(doc2VecModel, docIndex, num);
    const auto& doc = doc2VecModel.GetDocsHolder().GetDocument(docIndex);
    cout << "Document:" << endl;
    cout << '"' << doc->GetRawDocument() << '"' << endl << endl;

    if (similarDocs.empty()) {
        cout << "No similar documents were found." << endl << endl;
        return;
    }

    for (size_t i = 0; i < similarDocs.size(); ++i) {
        cout << "Similar document #" << i + 1 << ", similarity: " << similarDocs[i].Similarity << endl;
        cout << '"' << similarDocs[i].Document->GetRawDocument() << '"' << endl << endl;
    }
}

void FindAndPrintSimilarDocs(const TDoc2Vec& doc2VecModel, const std::string& docTag, unsigned int num) {
    const auto& docsHolder = doc2VecModel.GetDocsHolder();
    TDocument doc;
    if (!docsHolder.GetDocument(docTag, doc)) {
        cout << "No document with tag " << '"' << docTag << '"' << "." << endl;
        return;
    }
    FindAndPrintSimilarDocs(doc2VecModel, doc.GetIndex(), num);
}

void PrintWordVector(const TDoc2Vec& doc2VecModel, const std::string& word) {
    TWord wordStruct;
    const auto& wordsVoc = doc2VecModel.GetWordsVocabulary();
    const auto& neuralNetwork = doc2VecModel.GetNeuralNetwork();
    auto normWord = NormalizeWord(word);

    if (!wordsVoc.GetWord(normWord, wordStruct)) {
        cout << "Word " << '"' << word << '"' << " isn't in vocabulary." << endl;
        return;
    }

    const auto& wordVector = neuralNetwork.GetWordNormVector(wordStruct.Index);
    cout << "Vector for word " << '"' << word << '"' << ":" << endl;
    ostream_iterator<double> outIt(cout, " ");
    copy(wordVector.Begin(), wordVector.End(), outIt);
    cout << endl;
}

void PrintDocVector(const TDoc2Vec& doc2VecModel, const std::string& docTag) {
    const auto& docsHolder = doc2VecModel.GetDocsHolder();
    TDocument doc;
    if (!docsHolder.GetDocument(docTag, doc)) {
        cout << "No document with tag " << '"' << docTag << '"' << "." << endl;
        return;
    }

    const auto& neuralNetwork = doc2VecModel.GetNeuralNetwork();
    const auto& docVector = neuralNetwork.GetDocumentNormVector(doc.GetIndex());
    cout << "Vector for document " << '"' << docTag << '"' << ":" << endl;
    ostream_iterator<double> outIt(cout, " ");
    copy(docVector.Begin(), docVector.End(), outIt);
    cout << endl;
}
