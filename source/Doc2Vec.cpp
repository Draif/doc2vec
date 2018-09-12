#include "Doc2Vec.h"
#include "TrainThread.h"
#include "Common.h"

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;

vector<TTrainThreadSpec> TDoc2Vec::CreateThreadsSpecs() const {
    vector<TTrainThreadSpec> res;
    auto docsHolders = DocumentsHolder->SplitDocuments(Spec.ThreadCount);
    for (const auto docsHolder : docsHolders) {
        res.emplace_back(
            Spec,
            NeuralNetwork,
            WordsVocabulary,
            NegativeSampleTable,
            ExpTable,
            docsHolder
        );
    }
    assert(res.size() == Spec.ThreadCount);
    return res;
}

void TDoc2Vec::Train() {
    using namespace chrono;
    Spec.Print();
    auto threadsSpecs = CreateThreadsSpecs();
    cout << "Training started with " << Spec.ThreadCount << " threads." << endl;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Initial values for alpha
    Spec.Alpha->SetTotalTrainWords(Spec.IterationNumber * WordsVocabulary->GetTrainWordsCount());
    Spec.Alpha->StartCounting();

    vector<TTrainThread> trainThreadsObjects;
    vector<thread> threads;
    for (const auto& spec : threadsSpecs) {
        trainThreadsObjects.emplace_back(spec);
        threads.emplace_back(trainThreadsObjects.back());
    }

    for (auto& thread : threads)
        thread.join();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << endl << "Training ended and took " << time_span.count() << " seconds." << endl;

    NeuralNetwork->Normalize();
}

void TDoc2Vec::InitTables() {
    ExpTable = make_shared<vector<double>>(EXP_TABLE_SIZE, 0);
    for (size_t i = 0; i < ExpTable->size(); ++i) {
        (*ExpTable)[i] = exp((static_cast<double>(i) / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        (*ExpTable)[i] /= (*ExpTable)[i] + 1;
    }
    if (Spec.NegativeSampleNum > 0) {
        NegativeSampleTable = make_shared<vector<unsigned int>>(NEGATIVE_SAMPLE_TABLE_SIZE, 0);
        double power = 0.75;
        unsigned long long trainWordsPower = 0;
        for (auto it = WordsVocabulary->Begin(); it != WordsVocabulary->End(); ++it)
            trainWordsPower += static_cast<long long>(pow(it->second->Frequency, power));
        auto it = WordsVocabulary->Begin();
        unsigned int wordIndex = it->second->Index;
        double d1 = pow(it->second->Frequency, power) / trainWordsPower;
        for (size_t i = 0; i < NegativeSampleTable->size(); ++i) {
            (*NegativeSampleTable)[i] = wordIndex;
            if (static_cast<double>(i) / NEGATIVE_SAMPLE_TABLE_SIZE > d1) {
                if (it != WordsVocabulary->End()) {
                    ++it;
                    wordIndex = it->second->Index;
                }
                d1 += pow(it->second->Frequency, power) / trainWordsPower;
            }
        }
    }
}

string TTrainSpec::CLASS_TAG = "TTrainSpec";

void TTrainSpec::Save(std::ofstream& out) const {
    out << TTrainSpec::CLASS_TAG << endl;
    out << DimensionSize << SERIALIZE_DELIM << HierarchicalSoftmax << SERIALIZE_DELIM << CBOW << SERIALIZE_DELIM
        << NegativeSampleNum << SERIALIZE_DELIM << IterationNumber << SERIALIZE_DELIM << WindowSize << SERIALIZE_DELIM
        << Sample << SERIALIZE_DELIM << ThreadCount << SERIALIZE_DELIM
        << Alpha->Get() << endl;
    out << TrainFilename << endl;
    out << TTrainSpec::CLASS_TAG << endl;
}

void TTrainSpec::Load(std::ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TTrainSpec::CLASS_TAG)
        throw runtime_error("TTrainSpec::Load - wrong header.");
    in >> DimensionSize >> HierarchicalSoftmax >> CBOW >> NegativeSampleNum >> IterationNumber >> WindowSize >> Sample
        >> ThreadCount;
    double alpha;
    in >> alpha;
    Alpha = make_shared<TAlpha>(alpha);
    in >> TrainFilename;
    getline(in, buf);
    getline(in, buf);
    if (buf != TTrainSpec::CLASS_TAG)
        throw runtime_error("TTrainSpec::Load - wrong tail.");
}

string TDoc2Vec::CLASS_TAG = "TDoc2Vec";

void TDoc2Vec::Save(std::ofstream& out) const {
    cout << "Start to save model." << endl;
    const unsigned int maxSteps = 4;
    using namespace chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    out << TDoc2Vec::CLASS_TAG << endl;
    Spec.Save(out);
    PrintProgress(1, maxSteps);

    NeuralNetwork->Save(out);
    PrintProgress(2, maxSteps);

    DocumentsHolder->Save(out);
    PrintProgress(3, maxSteps);

    WordsVocabulary->Save(out);
    out << TDoc2Vec::CLASS_TAG << endl;
    PrintProgress(maxSteps, maxSteps);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << "Saving of model finished and took " << time_span.count() << " seconds." << endl;
}

void PrintProgress(unsigned int cur, unsigned int max) {
    const char success = 'X';
    const char wait = ' ';
    cout << '\r' << '[';
    for (size_t i = 1; i <= max; ++i) {
        if (cur >= i) {
            cout << success;
        } else {
            cout << wait;
        }
    }
    cout << ']' << flush;
    if (cur == max)
        cout << endl;
}

void TDoc2Vec::Load(std::ifstream& in) {
    cout << "Start to load model." << endl;
    const unsigned int maxSteps = 5;
    using namespace chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    PrintProgress(0, maxSteps);
    string buf;
    getline(in, buf);
    if (buf != TDoc2Vec::CLASS_TAG)
        throw runtime_error("TDoc2Vec::Load - wrong header.");

    Spec.Load(in);
    PrintProgress(1, maxSteps);

    TNeuralNetwork network;
    network.Load(in);
    NeuralNetwork = make_shared<TNeuralNetwork>(network);
    PrintProgress(2, maxSteps);

    TDocumentsHolder docsHolder;
    docsHolder.Load(in);
    DocumentsHolder = make_shared<TDocumentsHolder>(docsHolder);
    PrintProgress(3, maxSteps);

    TVocabulary voc;
    voc.Load(in);
    WordsVocabulary = make_shared<TVocabulary>(voc);
    PrintProgress(4, maxSteps);

    InitTables();

    getline(in, buf);
    if (buf != TDoc2Vec::CLASS_TAG)
        throw runtime_error("TDoc2Vec::Load - wrong tail.");

    PrintProgress(maxSteps, maxSteps);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << "Loading of model finished and took " << time_span.count() << " seconds." << endl;

    DocumentsHolder->PrintInfo();
    WordsVocabulary->PrintInfo("Words vocabulary");
    Spec.Print();
}
