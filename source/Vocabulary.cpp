#include "Vocabulary.h"

#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <memory>

using namespace std;

string NormalizeWord(const string& word) {
    string buf;
    buf.resize(word.size());
    transform(word.begin(), word.end(), buf.begin(), ::tolower);
    return buf;
}

void TVocabulary::BuildHuffmanTree() {
    vector<std::shared_ptr<TWord>> vocabulary(HashMap.size());
    size_t i = 0;
    for (auto& it : HashMap) {
        vocabulary[i] = it.second;
        ++i;
    }
    sort(vocabulary.begin(), vocabulary.end(),
        [](const std::shared_ptr<TWord> a, const std::shared_ptr<TWord> b){return *a < *b;}
    );

    size_t vectorSize = HashMap.size() * 2 + 1;
    vector<int> count(vectorSize), binary(vectorSize), parentNode(vectorSize);
    vector<int> code(MAX_CODE_LENGTH, false);
    vector<int> point(MAX_CODE_LENGTH, 0);

    for (size_t i = 0; i < vocabulary.size(); ++i)
        count[i] = vocabulary[i]->Frequency;
    for (size_t i = vocabulary.size(); i < 2 * vocabulary.size(); ++i)
        count[i] = numeric_limits<int>::max();

    long long min1i, min2i, pos1 = vocabulary.size() - 1, pos2 = vocabulary.size();
    for (size_t i = 0; i < vocabulary.size() - 1; ++i) {
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                --pos1;
            } else {
                min1i = pos2;
                ++pos2;
            }
        } else {
            min1i = pos2;
            ++pos2;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                --pos1;
            } else {
                min2i = pos2;
                ++pos2;
            }
        } else {
            min2i = pos2;
            ++pos2;
        }
        count[vocabulary.size() + i] = count[min1i] + count[min2i];
        parentNode[min1i] = vocabulary.size() + i;
        parentNode[min2i] = vocabulary.size() + i;
        binary[min2i] = 1;
    }
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        size_t b = i;
        size_t k = 0;
        while (true) {
            code[k] = binary[b];
            point[k] = b;
            k += 1;
            b = parentNode[b];
            if (b == vocabulary.size() * 2 - 2)
                break;
        }
        vocabulary[i]->Code.resize(k, 0);
        vocabulary[i]->Point.resize(k + 1, 0);
        vocabulary[i]->Point[0] = vocabulary.size() - 2;
        for (b = 0; b < k; ++b) {
            vocabulary[i]->Code[k - b - 1] = code[b];
            vocabulary[i]->Point[k - b] = point[b] - vocabulary.size();
        }
    }
}

string TWord::CLASS_TAG = "TWord";

void TWord::Save(ofstream& out) const {
    out << CLASS_TAG << endl;
    out << Frequency << SERIALIZE_DELIM << Index << endl;
    out << Word << endl;
    out << Point.size();
    for (const auto& it : Point)
        out << SERIALIZE_DELIM << it;
    out << endl;
    out << Code.size();
    for (const auto& it : Code)
        out << SERIALIZE_DELIM << it;
    out << endl;
    out << CLASS_TAG << endl;
}

void TWord::Load(ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TWord::CLASS_TAG)
        throw runtime_error("TWord::Load - wrong header.");

    in >> Frequency >> Index;
    in >> Word;
    unsigned int size;
    int tmp;
    in >> size;
    for (size_t i = 0; i < size; ++i) {
        in >> tmp;
        Point.push_back(tmp);
    }
    in >> size;
    for (size_t i = 0; i < size; ++i) {
        in >> tmp;
        Code.push_back(tmp);
    }

    getline(in, buf);
    getline(in, buf);
    if (buf != TWord::CLASS_TAG)
        throw runtime_error("TWord::Load - wrong tail.");
}

string TVocabulary::CLASS_TAG = "TVocabulary";

void TVocabulary::Save(ofstream& out) const {
    out << CLASS_TAG << endl;
    out << HashMap.size() << SERIALIZE_DELIM <<  IndexCounter << SERIALIZE_DELIM << TrainWordsCount << endl;
    for (const auto& it : HashMap)
        it.second->Save(out);
    out << CLASS_TAG << endl;
}

void TVocabulary::Load(ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TVocabulary::CLASS_TAG)
        throw runtime_error("TVocabulary::Load - wrong header.");
    unsigned int size;
    in >> size >> IndexCounter >> TrainWordsCount;
    getline(in, buf);
    for (size_t i = 0; i < size; ++i) {
        TWord word;
        word.Load(in);
        std::shared_ptr<TWord> ptr = std::make_shared<TWord>(word);
        HashMap[word.Word] = ptr;
        HashMapIdToWord[word.Index] = ptr;
    }
    getline(in, buf);
    if (buf != TVocabulary::CLASS_TAG)
        throw runtime_error("TVocabulary::Load - wrong tail.");
}

string TDocument::CLASS_TAG = "TDocument";

void TDocument::Save(std::ofstream& out) const {
    out << CLASS_TAG << endl;
    out << Index << endl;
    out << RawDocument << endl;
    out << CLASS_TAG << endl;
}

void TDocument::Load(std::ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TDocument::CLASS_TAG)
        throw runtime_error("TDocument::Load - wrong header.");

    in >> Index;
    getline(in, buf);
    getline(in, RawDocument);
    BuildFromRawDocument(RawDocument);

    getline(in, buf);
    if (buf != TDocument::CLASS_TAG)
        throw runtime_error("TDocument::Load - wrong tail.");
}

string TDocumentsHolder::CLASS_TAG = "TDocumentsHolder";

void TDocumentsHolder::Save(std::ofstream& out) const {
    out << TDocumentsHolder::CLASS_TAG << endl;
    out << Documents.size() << endl;
    for (const auto& doc : Documents)
        doc->Save(out);
    out << TDocumentsHolder::CLASS_TAG << endl;
}

void TDocumentsHolder::Load(std::ifstream& in) {
    string buf;
    getline(in, buf);
    if (buf != TDocumentsHolder::CLASS_TAG)
        throw runtime_error("TDocumentsHolder::Load - wrong header.");
    unsigned int size;
    in >> size;
    getline(in, buf);
    for (size_t i = 0; i < size; ++i) {
        TDocument doc;
        doc.Load(in);
        Documents.emplace_back(make_shared<TDocument>(doc));
        DocTagToIndex[doc.GetTag()] = doc.GetIndex();
    }
    getline(in, buf);
    if (buf != TDocumentsHolder::CLASS_TAG)
        throw runtime_error("TDocumentsHolder::Load - wrong tail.");
}
