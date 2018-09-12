#pragma once
#include "NeuralNetwork.h"
#include "Vocabulary.h"
#include "Common.h"
#include "Doc2Vec.h"

#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <chrono>


struct TDocumentTrainContext {
    TDocumentTrainContext()
        : Valid(false)
    {}

    TLayerVector<double>* DocumentVector;
    unsigned int SentenceLength;
    unsigned int SentenceNosampleLength;
    std::vector<unsigned int> SentenceNosample;
    std::vector<unsigned int> Sentence;
    bool Valid;
};

class TTrainThread {
public:
    TTrainThread(const TTrainThreadSpec& spec)
        : Spec(spec)
        , Distribution(0.0, 1.0)
        , WordCount(0)
    {}

    void operator()() {
        for (unsigned int iter = 0; iter < Spec.IterationNumber; ++iter) {
            for (const auto& doc : Spec.DocumentsHolder.GetDocuments()) {
                if (WordCount > UPDATE_WORD_NUMBER) {
                    Spec.Alpha->Update(WordCount); // Update learning rate
                    WordCount = 0;
                }
                TDocumentTrainContext docContext = BuildDocument(*doc);
                if (!docContext.Valid)
                    continue;
                TrainDocument(docContext);
            }
            Spec.Alpha->Update(WordCount);
            WordCount = 0;
        }
    }
private:
    TDocumentTrainContext BuildDocument(const TDocument& doc);
    void TrainDocument(const TDocumentTrainContext& docContext);
    void TrainSampleCBOW(unsigned int, const std::vector<unsigned int>&, TLayerVector<double>&);
    void TrainSampleSG(const std::vector<unsigned int>&);
    void TrainPairSG(unsigned int lastWord, TLayerVector<double>& DocumentVector);

private:
    bool DownSample(unsigned int wordFrequency) {
        if (Spec.Sample > 0) {
            auto tmp = Spec.Sample * Spec.WordsVocabulary->GetTrainWordsCount();
            double ran = (std::sqrt(wordFrequency / tmp) + 1) * tmp / wordFrequency;
            return ran < Distribution(RandGenerator);
        }
        return false;
    }

    unsigned int ChooseNegativeSample() const {
        size_t randIndex = rand() % Spec.NegativeSampleTable->size();
        return (*Spec.NegativeSampleTable)[randIndex];
    }

    double GetExpTableCell(double f) const {
        size_t index = static_cast<size_t>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
        assert(index < Spec.ExpTable->size());
        return (*Spec.ExpTable)[index];
    }
private:
    TTrainThreadSpec Spec;
    std::default_random_engine RandGenerator;
    std::uniform_real_distribution<double> Distribution;
    unsigned long long WordCount;
};
