#pragma once
#include "NeuralNetwork.h"
#include "Vocabulary.h"
#include "Common.h"

#include <string>
#include <memory>
#include <atomic>
#include <iostream>
#include <chrono>
#include <mutex>

void PrintProgress(unsigned int cur, unsigned int max);

class TAlpha {
public:
    TAlpha(double init)
        : InitialValue(init)
        , CurrentValue(init)
        , WordCountActual(0)
        , LastWordCount(0)
        , TotalTrainWords(0)
    {}

    TAlpha(const TAlpha& another)
        : InitialValue(another.InitialValue)
        , CurrentValue(another.CurrentValue)
        , LastWordCount(another.LastWordCount)
        , TotalTrainWords(another.TotalTrainWords)
    {}

    double Get() {
        return CurrentValue;
    }

    void Update(unsigned long long processedWordCount) {
        WordCountActual += processedWordCount;

        if (WordCountActual - LastWordCount > UPDATE_WORD_NUMBER) {
            if (!Mutex.try_lock())
                return;
            LastWordCount = WordCountActual;
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - TimePoint);
            double wordsBySec = static_cast<double>(WordCountActual) / time_span.count() / 1000;
            double progress = static_cast<double>(WordCountActual) / (TotalTrainWords);
            double updatedAlpha = InitialValue * (1 - progress);
            if (updatedAlpha > InitialValue  * ALPHA_MAX_REDUCE_COEFFICENT)
                CurrentValue = updatedAlpha;

            std::cout << "\r" << "Alpha: " << CurrentValue << " Progress: " << progress * 100
                << "% Words/Sec: " << wordsBySec << "k"<< std::flush;
            Mutex.unlock();
        }
    }

    void StartCounting() {
        TimePoint = std::chrono::high_resolution_clock::now();
    }

    void SetTotalTrainWords(unsigned long long totalTrainWords) {
        TotalTrainWords = totalTrainWords;
    }
private:
    const double InitialValue;
    double CurrentValue;
    std::atomic<unsigned long long> WordCountActual;
    unsigned long long LastWordCount;
    unsigned long long TotalTrainWords;
    std::mutex Mutex;
    std::chrono::high_resolution_clock::time_point TimePoint;
};

struct TTrainSpec {
    TTrainSpec()
        : DimensionSize(DEFAULT_DIMENSION_SIZE)
        , HierarchicalSoftmax(DEFAULT_HIERARCHICAL_SOFTMAX)
        , CBOW(DEFAULT_CBOW)
        , NegativeSampleNum(DEFAULT_NEGATIVE_SAMPLE_NUMBER)
        , IterationNumber(DEFAULT_ITERATION_NUMBER)
        , WindowSize(DEFAULT_WINDOW_SIZE)
        , Sample(DEFAULT_SAMPLE)
        , ThreadCount(DEFAULT_THREAD_COUNT)
        , Alpha(new TAlpha(DEFAULT_ALPHA))
    {}

    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);

    void Print() const {
        std::cout << "Training specs:" << std::endl
            << '\t' << "DimensionSize: " << DimensionSize << std::endl
            << '\t' << "HierarchicalSoftmax: " << HierarchicalSoftmax << std::endl
            << '\t' << "CBOW: " << CBOW << std::endl
            << '\t' << "NegativeSampleNum: " << NegativeSampleNum << std::endl
            << '\t' << "IterationNumber: " << IterationNumber << std::endl
            << '\t' << "WindowSize: " << WindowSize << std::endl
            << '\t' << "Sample: " << Sample << std::endl
            << '\t' << "ThreadCount: " << ThreadCount << std::endl
            << '\t' << "Alpha: " << Alpha->Get() << std::endl
            << '\t' << "Dataset filename: " << TrainFilename << std::endl;
    }

public:
    unsigned int DimensionSize;
    bool HierarchicalSoftmax;
    bool CBOW;
    int NegativeSampleNum;
    unsigned int IterationNumber;
    unsigned int WindowSize;
    double Sample;
    unsigned int ThreadCount;
    std::string TrainFilename;
    std::shared_ptr<TAlpha> Alpha;

    static std::string CLASS_TAG;
};

struct TTrainThreadSpec {
    TTrainThreadSpec(
        const TTrainSpec& Spec,
        const std::shared_ptr<TNeuralNetwork>& neuralNetwork,
        const std::shared_ptr<TVocabulary>& wordsVocabulary,
        const std::shared_ptr<std::vector<unsigned int>>& negativeSampleTable,
        const std::shared_ptr<std::vector<double>>& expTable,
        const TDocumentsHolder& documentsHolder
    )
        : IterationNumber(Spec.IterationNumber)
        , Alpha(Spec.Alpha)
        , WindowSize(Spec.WindowSize)
        , HierarchicalSoftmax(Spec.HierarchicalSoftmax)
        , CBOW(Spec.CBOW)
        , NegativeSampleNum(Spec.NegativeSampleNum)
        , DimensionSize(Spec.DimensionSize)
        , Sample(Spec.Sample)
        , NeuralNetwork(neuralNetwork)
        , WordsVocabulary(wordsVocabulary)
        , NegativeSampleTable(negativeSampleTable)
        , ExpTable(expTable)
        , DocumentsHolder(documentsHolder)
    {}

public:
    unsigned int IterationNumber;
    std::shared_ptr<TAlpha> Alpha;
    unsigned int WindowSize;
    bool HierarchicalSoftmax;
    bool CBOW;
    unsigned int NegativeSampleNum;
    unsigned int DimensionSize;
    double Sample;
    std::shared_ptr<TNeuralNetwork> NeuralNetwork;
    std::shared_ptr<TVocabulary> WordsVocabulary;
    std::shared_ptr<std::vector<unsigned int>> NegativeSampleTable;
    std::shared_ptr<std::vector<double>> ExpTable;
    TDocumentsHolder DocumentsHolder;
};

class TDoc2Vec {
public:
    TDoc2Vec() {}

    TDoc2Vec(const TTrainSpec& spec)
        : Spec(spec)
    {
        std::cout << "Model creation started." << std::endl;
        unsigned int maxSteps = 4;
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        PrintProgress(0, maxSteps);
        DocumentsHolder = std::make_shared<TDocumentsHolder>(Spec.TrainFilename);
        PrintProgress(1, maxSteps);
        WordsVocabulary = std::make_shared<TVocabulary>(DocumentsHolder->CreateWordsVocabulary());
        PrintProgress(2, maxSteps);
        NeuralNetwork = std::make_shared<TNeuralNetwork>(
            WordsVocabulary->GetSize(),
            DocumentsHolder->GetSize(),
            Spec.DimensionSize
        );
        PrintProgress(3, maxSteps);
        InitTables();
        PrintProgress(maxSteps, maxSteps);

        DocumentsHolder->PrintInfo();
        WordsVocabulary->PrintInfo("Words vocabulary");

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        std::cout << "Model creation finished and took " << time_span.count() << " seconds." << std::endl;
    }

    void Train();

    const TNeuralNetwork& GetNeuralNetwork() const {
        return *NeuralNetwork;
    }

    const TVocabulary& GetWordsVocabulary() const {
        return *WordsVocabulary;
    }

    const TDocumentsHolder& GetDocsHolder() const {
        return *DocumentsHolder;
    }

    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);

private:
    void InitTables();
    std::vector<TTrainThreadSpec> CreateThreadsSpecs() const;
private:
    TTrainSpec Spec;
    std::shared_ptr<TNeuralNetwork> NeuralNetwork;
    std::shared_ptr<TDocumentsHolder> DocumentsHolder;
    std::shared_ptr<TVocabulary> WordsVocabulary;
    std::shared_ptr<std::vector<double>> ExpTable;
    std::shared_ptr<std::vector<unsigned int>> NegativeSampleTable;

    static std::string CLASS_TAG;
};
