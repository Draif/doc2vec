#include "TrainThread.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include <chrono>
#include <iostream>
#include <cmath>

using namespace std;

TDocumentTrainContext TTrainThread::BuildDocument(const TDocument& doc) {
    TDocumentTrainContext Context;
    Context.DocumentVector = &Spec.NeuralNetwork->GetDocumentVector(doc.GetIndex());
    for (const auto& wordStr : doc.GetWords()) {
        shared_ptr<TWord> word;
        if (!Spec.WordsVocabulary->GetWord(wordStr, word))
            continue;
        WordCount += 1;
        Context.SentenceNosample.push_back(word->Index);
        if (!DownSample(word->Frequency)) {
            Context.Sentence.push_back(word->Index);
        }
    }
    Context.Valid = true;
    return Context;
}

void TTrainThread::TrainDocument(const TDocumentTrainContext& docContext) {
    size_t sentenceSize = docContext.Sentence.size();
    int sentenceSizeInt = static_cast<int>(sentenceSize);
    int windowSize = static_cast<int>(Spec.WindowSize);
    for (size_t sentencePosition = 0; sentencePosition < sentenceSize; ++sentencePosition) {
        std::vector<unsigned int> context;
        int b = rand() % Spec.WindowSize;
        int sentencePositionInt = static_cast<int>(sentencePosition);
        size_t contextStart = static_cast<size_t>(std::max(0, sentencePositionInt - windowSize + b));
        size_t contextEnd = static_cast<size_t>(std::min(sentenceSizeInt, sentencePositionInt + windowSize - b + 1));
        for (size_t i = contextStart; i < contextEnd; ++i) {
            if (i != sentencePosition)
                context.push_back(docContext.Sentence[i]);
        }

        if (Spec.CBOW) {
            TrainSampleCBOW(docContext.Sentence[sentencePosition], context, *docContext.DocumentVector);
        } else {
            TrainSampleSG(context);
        }
    }

    if (!Spec.CBOW) {
        for (const auto& lastWord : docContext.SentenceNosample) {
            TrainPairSG(lastWord, *docContext.DocumentVector);
        }
    }
}

void TTrainThread::TrainSampleSG(const std::vector<unsigned int>& context) {
    for (const auto& lastWord : context) {
        auto& vector = Spec.NeuralNetwork->GetWordVector(lastWord);
        TrainPairSG(lastWord, vector);
    }
}

void TTrainThread::TrainPairSG(unsigned int centralWord, TLayerVector<double>& context) {
    TSimpleLockGuard<TLayerVector<double>> lgContext(context);

    vector<double> Neu1E(Spec.DimensionSize, 0);
    if (Spec.HierarchicalSoftmax) {
        shared_ptr<TWord> word;
        if (!Spec.WordsVocabulary->GetWord(centralWord, word))
            throw runtime_error("TrainPairSG: HierarchicalSoftmax can't find word");
        for (size_t d = 0; d < word->Code.size(); ++d) {
            double f = 0;
            size_t wordIndex = word->Point[d];

            TLayerVector<double>& wordVector = Spec.NeuralNetwork->GetHierarchicalSoftmaxVector(wordIndex);
            TSimpleLockGuard<TLayerVector<double>> lgWord(wordVector);

            // hidden -> output
            assert(context.Size() == wordVector.Size());
            for (size_t i = 0; i < context.Size(); ++i)
                f += context[i] * wordVector[i];

            if (std::isnan(f) || f <= -MAX_EXP || f >= MAX_EXP) {
                continue;
            } else {
                f = GetExpTableCell(f);
            }

            // gradient
            double g = (1.0 - static_cast<double>(word->Code[d]) - f) * Spec.Alpha->Get();

            // output -> hidden
            assert(Neu1E.size() == wordVector.Size());
            for (size_t i = 0; i < Neu1E.size(); ++i)
                Neu1E[i] += g * wordVector[i];

            // learn weights
            for (size_t i = 0; i < context.Size(); ++i)
                wordVector[i] += g * context[i];
        }
    }

    if (Spec.NegativeSampleNum > 0) {
        for (size_t d = 0; d <= Spec.NegativeSampleNum; ++d) {
            unsigned int target;
            double label;
            if (d == 0) {
                target = centralWord;
                label = 1;
            } else {
                target = ChooseNegativeSample();
                if (target == centralWord)
                    continue;
                label = 0;
            }

            double f = 0, g = 0;
            TLayerVector<double>& negativeSampleVector = Spec.NeuralNetwork->GetNegativeSampleVector(target);
            TSimpleLockGuard<TLayerVector<double>> lgNeg(negativeSampleVector);;

            assert(negativeSampleVector.Size() == context.Size());
            assert(negativeSampleVector.Size() == Spec.DimensionSize);
            for (size_t i = 0; i < negativeSampleVector.Size(); ++i)
                f += context[i] * negativeSampleVector[i];

            if (f > MAX_EXP) {
                g = (label - 1) * Spec.Alpha->Get();
            } else if (f < -MAX_EXP) {
                g = label * Spec.Alpha->Get();
            } else {
                g = (label - GetExpTableCell(f)) * Spec.Alpha->Get();
            }

            assert(Neu1E.size() == negativeSampleVector.Size());
            assert(Neu1E.size() == Spec.DimensionSize);
            for (size_t i = 0; i < Neu1E.size(); ++i)
                Neu1E[i] += g * negativeSampleVector[i];

            for (size_t i = 0; i < context.Size(); ++i)
                negativeSampleVector[i] += g * context[i];
        }
    }

    // input -> hidden
    assert(Neu1E.size() == context.Size());
    for (size_t i = 0; i < Neu1E.size(); ++i)
        context[i] += Neu1E[i];
}

void TTrainThread::TrainSampleCBOW(
    unsigned int centralWord,
    const vector<unsigned int>& context,
    TLayerVector<double>& docVector
) {
    TSimpleLockGuard<TLayerVector<double>> lgDoc(docVector);

    vector<double> Neu1(Spec.DimensionSize, 0);
    vector<double> Neu1E(Spec.DimensionSize, 0);
    unsigned int cw = 0;

    // in -> Hidden
    for (const auto& contextIndex : context) {
        auto& wordVector = Spec.NeuralNetwork->GetWordVector(contextIndex);
        TSimpleLockGuard<TLayerVector<double>> lgWord(wordVector);

        assert(wordVector.Size() == Neu1.size());
        for (size_t i = 0; i < Neu1.size(); ++i)
            Neu1[i] += wordVector[i];
        cw += 1;
    }

    assert(docVector.Size() == Neu1.size());
    for (size_t i = 0; i < Neu1.size(); ++i)
        Neu1[i] += docVector[i];

    cw += 1;
    for (size_t i = 0; i < Neu1.size(); ++i)
        Neu1[i] /= cw;


    if (Spec.HierarchicalSoftmax) {
        shared_ptr<TWord> word;
        if (!Spec.WordsVocabulary->GetWord(centralWord, word))
            throw runtime_error("TrainSampleCBOW: HierarchicalSoftmax can't find word");
        for (size_t d = 0; d < word->Code.size(); ++d) {
            double f = 0;
            size_t wordIndex = word->Point[d];

            TLayerVector<double>& wordVector = Spec.NeuralNetwork->GetHierarchicalSoftmaxVector(wordIndex);
            TSimpleLockGuard<TLayerVector<double>> lgWord(wordVector);

            // hidden -> output
            assert(Neu1.size() == wordVector.Size());
            for (size_t i = 0; i < Neu1.size(); ++i)
                f += Neu1[i] * wordVector[i];

            if (std::isnan(f) || f <= -MAX_EXP || f >= MAX_EXP) {
                continue;
            } else {
                f = GetExpTableCell(f);
            }

            // gradient
            double g = (1.0 - static_cast<double>(word->Code[d]) - f) * Spec.Alpha->Get();

            // output -> hidden
            assert(Neu1E.size() == wordVector.Size());
            for (size_t i = 0; i < Neu1E.size(); ++i)
                Neu1E[i] += g * wordVector[i];

            // learn weights
            for (size_t i = 0; i < Neu1E.size(); ++i)
                wordVector[i] += g * Neu1[i];
        }
    }

    if (Spec.NegativeSampleNum > 0) {
        for (size_t d = 0; d <= Spec.NegativeSampleNum; ++d) {
            unsigned int target;
            double label;
            if (d == 0) {
                target = centralWord;
                label = 1;
            } else {
                target = ChooseNegativeSample();
                if (target == centralWord)
                    continue;
                label = 0;
            }

            double f = 0, g = 0;
            TLayerVector<double>& negativeSampleVector = Spec.NeuralNetwork->GetNegativeSampleVector(target);
            TSimpleLockGuard<TLayerVector<double>> lgNeg(negativeSampleVector);

            assert(negativeSampleVector.Size() == Neu1.size());
            assert(Neu1.size() == Spec.DimensionSize);
            for (size_t i = 0; i < negativeSampleVector.Size(); ++i)
                f += Neu1[i] * negativeSampleVector[i];

            if (std::isnan(f) || f > MAX_EXP) {
                g = (label - 1) * Spec.Alpha->Get();
            } else if (f < -MAX_EXP) {
                g = label * Spec.Alpha->Get();
            } else {
                g = (label - GetExpTableCell(f)) * Spec.Alpha->Get();
            }

            assert(Neu1E.size() == negativeSampleVector.Size());
            assert(Neu1E.size() == Spec.DimensionSize);
            for (size_t i = 0; i < Neu1E.size(); ++i)
                Neu1E[i] += g * negativeSampleVector[i];

            for (size_t i = 0; i < Neu1.size(); ++i)
                negativeSampleVector[i] += g * Neu1[i];
        }
    }

    // hidden -> in
    for (const auto& lastWord : context) {
        TLayerVector<double>& wordVector = Spec.NeuralNetwork->GetWordVector(lastWord);
        TSimpleLockGuard<TLayerVector<double>> lgWord(wordVector);

        assert(wordVector.Size() == Neu1E.size());
        for (size_t i = 0; i < Neu1E.size(); ++i)
            wordVector[i] += Neu1E[i];
    }

    assert(docVector.Size() == Neu1E.size());
    for (size_t i = 0; i < Neu1E.size(); ++i)
        docVector[i] += Neu1E[i];
}
