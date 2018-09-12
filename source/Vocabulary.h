#pragma once
#include "Common.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <regex>
#include <iostream>

std::string NormalizeWord(const std::string& word);

struct TWord {
    TWord() {}
    TWord(const std::string word, unsigned int index)
        : Word(word)
        , Frequency(1)
        , Index(index)
    {}
    TWord(const TWord& another)
        : Word(another.Word)
        , Frequency(another.Frequency)
        , Index(another.Index)
        , Point(another.Point)
        , Code(another.Code)
    {}

public:
    std::string Word;
    unsigned int Frequency;
    unsigned int Index;
    std::vector<int> Point;
    std::vector<int> Code;
public:
    bool operator<(const TWord& another) const {
        return Frequency < another.Frequency;
    }

    void DebugPrint() const {
        std::cout << "Word[" << Word << "] Index[" << Index << "] Frequency[" << Frequency
            << "] Code[";
        for (const auto& codeIt : Code)
            std::cout << static_cast<int>(codeIt) << "-";
        std::cout << "]" << std::endl;
    }

    void Save(std::ofstream&) const;
    void Load(std::ifstream&);
private:
    static std::string CLASS_TAG;
};

class TVocabulary {
public:
    TVocabulary()
        : IndexCounter(0)
        , TrainWordsCount(0)
    {};

    ~TVocabulary() {};

    void AddWord(const std::string& word) {
        auto normWord = NormalizeWord(word);
        auto wordIt = HashMap.find(normWord);
        if (wordIt != HashMap.end()) {
            wordIt->second->Frequency += 1;
        } else {
            std::shared_ptr<TWord> ptr = std::make_shared<TWord>(normWord, IndexCounter);
            HashMap[normWord] = ptr;
            HashMapIdToWord[IndexCounter] = ptr;
            IndexCounter += 1;
        }
        TrainWordsCount += 1; // TODO: maybe wrong here
    }

    bool GetWord(const std::string& word, std::shared_ptr<TWord>& res) const {
        auto wordIt = HashMap.find(word);
        if (wordIt != HashMap.end()) {
            res = wordIt->second;
            return true;
        }
        return false;
    }

    bool GetWord(const std::string& word, TWord& res) const {
        auto wordIt = HashMap.find(word);
        if (wordIt != HashMap.end()) {
            res = *wordIt->second;
            return true;
        }
        return false;
    }

    bool GetWord(unsigned int wordIndex, std::shared_ptr<TWord>& res) const {
        auto wordIt = HashMapIdToWord.find(wordIndex);
        if (wordIt != HashMapIdToWord.end()) {
            res = wordIt->second;
            return true;
        }
        return false;
    }

    bool GetWord(unsigned int wordIndex, TWord& res) const {
        auto wordIt = HashMapIdToWord.find(wordIndex);
        if (wordIt != HashMapIdToWord.end()) {
            res = *wordIt->second;
            return true;
        }
        return false;
    }

    unsigned int GetTrainWordsCount() const {
        return TrainWordsCount;
    }

    size_t GetSize() const {
        return HashMap.size();
    }

    std::unordered_map<std::string, std::shared_ptr<TWord>>::const_iterator Begin() const {
        return HashMap.begin();
    }

    std::unordered_map<std::string, std::shared_ptr<TWord>>::const_iterator End() const {
        return HashMap.end();
    }


    void PrintInfo(const std::string vocName) const {
        std::cout << "Vocabulary [" << vocName << "] was built." << std::endl
        << "Statistics:" << std::endl
        << "\t" << HashMap.size() << " unique words." << std::endl
        << "\t" << TrainWordsCount << " train words." << std::endl;
    }

public:
    void BuildHuffmanTree();
    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);
private:
    std::unordered_map<std::string, std::shared_ptr<TWord>> HashMap;
    std::unordered_map<unsigned int, std::shared_ptr<TWord>> HashMapIdToWord;
    unsigned int IndexCounter;
    unsigned int TrainWordsCount;
private:
    static std::string CLASS_TAG;
};

class TDocument {
public:
    TDocument() {}

    TDocument(const std::string input, unsigned int index)
        : RawDocument(input)
        , Index(index)
    {
        BuildFromRawDocument(input);
    }

    const std::vector<std::string>& GetWords() const {
        return Words;
    }

    const std::string& GetTag() const {
        return Tag;
    }

    const std::string& GetRawDocument() const {
        return RawDocument;
    }

    unsigned int GetIndex() const {
        return Index;
    }

    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);
private:
    void BuildFromRawDocument(const std::string& input) {
        std::regex reg("\\w+");
        size_t firstSpace = input.find(" ");
		Tag = std::string(input.begin(), input.begin() + firstSpace);
        for(std::sregex_iterator it(input.begin() + firstSpace + 1, input.end(), reg), it_end; it != it_end; ++it)
            Words.push_back((*it)[0]);
    }

    std::vector<std::string> Words;
    std::string Tag;
    std::string RawDocument;
    unsigned int Index;
private:
    static std::string CLASS_TAG;
};

class TDocumentsHolder {
public:
    TDocumentsHolder() {}

    TDocumentsHolder(const std::string filename) {
        std::ifstream file(filename);
        std::string line;
        while(std::getline(file, line)) {
            unsigned int docIndex = Documents.size();
            Documents.emplace_back(std::make_shared<TDocument>(line, docIndex));

            const auto& docTag = Documents.back()->GetTag();
            if (DocTagToIndex.count(docTag))
                throw std::runtime_error("There are several documents with same tag");

            DocTagToIndex[docTag] = docIndex;
        }

        if (Documents.empty())
            throw std::runtime_error("No documents in dataset file");
    }

    TDocumentsHolder(const std::vector<std::shared_ptr<TDocument>>& docVector)
        : Documents(docVector)
    {}

    std::vector<TDocumentsHolder> SplitDocuments(unsigned int parts) const {
        std::vector<TDocumentsHolder> docsHolders;
        unsigned int numDocsInPart = Documents.size() / parts + 1;
        auto it = Documents.begin();
        for (int i = 0, iMax = static_cast<int>(Documents.size()) - numDocsInPart;
                i < iMax;
                i += numDocsInPart, it += numDocsInPart
        ) {
            std::vector<std::shared_ptr<TDocument>> tmpVector(it, it + numDocsInPart);
            docsHolders.emplace_back(tmpVector);
        }
        std::vector<std::shared_ptr<TDocument>> tmpVector(it, Documents.end());
        docsHolders.emplace_back(tmpVector);
        return docsHolders;
    }

    TVocabulary CreateWordsVocabulary() const {
        TVocabulary vocabulary;
        for (const auto& doc : Documents) {
            for (const auto& word : doc->GetWords())
                vocabulary.AddWord(word);
        }
        vocabulary.BuildHuffmanTree();
        return vocabulary;
    }

    unsigned int GetSize() const {
        return Documents.size();
    }

    const std::vector<std::shared_ptr<TDocument>>& GetDocuments() const {
        return Documents;
    }

    const std::shared_ptr<TDocument>& GetDocument(unsigned int docIndex) const {
        if (docIndex >= Documents.size())
            throw std::runtime_error("GetDocument - out of range");
        return Documents[docIndex];
    }

    bool GetDocument(const std::string& docTag, TDocument& docRes) const {
        if (!DocTagToIndex.count(docTag))
            return false;

        const auto& docPtr = GetDocument(DocTagToIndex.at(docTag));
        docRes = *docPtr;
        return true;
    }

    void PrintInfo() const {
        std::cout << "Documents holder was built." << std::endl
        << "\t" << Documents.size() << " documents" << std::endl;
    }

    void Save(std::ofstream& out) const;
    void Load(std::ifstream& in);
private:
	std::vector<std::shared_ptr<TDocument>> Documents;
    std::unordered_map<std::string, unsigned int> DocTagToIndex;

    static std::string CLASS_TAG;
};
