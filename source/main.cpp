#include "Doc2Vec.h"
#include "Algorithm.h"

using namespace std;

TDoc2Vec LoadModel(const string& filename) {
    TDoc2Vec model;
    ifstream ifs(filename);

    if (ifs.is_open()) {
        model.Load(ifs);
        ifs.close();
    } else {
        throw runtime_error("Cannot open file <" + filename + ">.");
    }

    return model;
}

void SaveModel(const TDoc2Vec& model, const string& filename) {
    ofstream ofs(filename);
    if (ofs.is_open()) {
        model.Save(ofs);
        ofs.close();
    } else {
        throw runtime_error("Cannot open file <" + filename + ">.");
    }
}

char* GetCmdOption(char** begin, char** end, const string& option) {
    char** itr = find(begin, end, option);
    if (itr != end && ++itr != end)
        return *itr;
    return 0;
}

vector<string> GetAllCmdOptions(char** begin, char** end, const string& option) {
    vector<string> res;
    char** itr = begin;
    while (itr != end) {
        itr = find(itr, end, option);
        if (itr != end && ++itr != end)
            res.emplace_back(*itr);
    }
    return res;
}

bool CmdOptionExists(char** begin, char** end, const string& option) {
    return std::find(begin, end, option) != end;
}

template<typename T>
bool GetAndSaveOption(char** begin, char** end, const string& option, T& res, bool enableZero=false) {
    char* resStr = GetCmdOption(begin, end, option);
    if (resStr) {
        int resNum = atoi(resStr);
        if (resNum <= 0) {
            if (resNum != 0 || !enableZero) {
                cerr << "Option " << option << " should have positive value greater than 0." << endl;
                return false;
            }
        }
        res = resNum;
    }
    return true;
}

template<>
bool GetAndSaveOption<double>(char** begin, char** end, const string& option, double& res, bool enableZero) {
    char* resStr = GetCmdOption(begin, end, option);
    if (resStr) {
        double resNum = atof(resStr);
        if (resNum <= 0) {
            if (resNum != 0 || !enableZero) {
                cerr << "Option " << option << " should have positive value greater than 0." << endl;
                return false;
            }
        }
        res = resNum;
    }
    return true;
}

int Train(int argc, char* argv[]) {
    char** begin = argv + 2;
    char** end = argv + argc;
    TTrainSpec Spec;

    char* datasetFile = GetCmdOption(begin, end, DATA_OPTION);
    if (!datasetFile) {
        cerr << "Need to specify filename of dataset with option " << DATA_OPTION << "." << endl;
        return FAIL_RETURN;
    }
    Spec.TrainFilename = datasetFile;

    if (!(GetAndSaveOption(begin, end, DIMENSION_OPTION, Spec.DimensionSize)
        && GetAndSaveOption(begin, end, ITER_OPTION, Spec.IterationNumber)
        && GetAndSaveOption(begin, end, WINDOW_OPTION, Spec.WindowSize)
        && GetAndSaveOption(begin, end, THREAD_OPTION, Spec.ThreadCount)
        && GetAndSaveOption(begin, end, NS_NUM_OPTION, Spec.NegativeSampleNum, /*enableZero*/ true)
        && GetAndSaveOption<double>(begin, end, SAMPLE_OPTION, Spec.Sample, false)
    ))
        return FAIL_RETURN;

    if (CmdOptionExists(begin, end, HS_OPTION))
        Spec.HierarchicalSoftmax = true;
    if (CmdOptionExists(begin, end, NO_CBOW_OPTION))
        Spec.CBOW = false;

    char* resStr = GetCmdOption(begin, end, ALPHA_OPTION);
    if (resStr) {
        double resNum = atof(resStr);
        if (resNum <= 0) {
            cerr << "Option " << ALPHA_OPTION << " should have positive value greater than 0." << endl;
            return FAIL_RETURN;
        }
        if (Spec.Alpha->Get() != resNum)
            Spec.Alpha = make_shared<TAlpha>(resNum);
    }

    TDoc2Vec model(Spec);
    model.Train();

    char* filenameSave = GetCmdOption(begin, end, SAVE_OPTION);
    if (filenameSave)
        SaveModel(model, filenameSave);

    return SUCCESS_RETURN;
}

int Similar(int argc, char* argv[]) {
    char** begin = argv + 2;
    char** end = argv + argc;

    char* filename = GetCmdOption(begin, end, LOAD_OPTION);
    if (!filename) {
        cerr << "Need to specify saved model filename with option " << LOAD_OPTION << "." << endl;
        return FAIL_RETURN;
    }

    vector<string> words = GetAllCmdOptions(begin, end, WORD_OPTION);
    vector<string> docs = GetAllCmdOptions(begin, end, DOC_OPTION);

    if (words.empty() && docs.empty()) {
        cerr << "Need to specify either " << WORD_OPTION << " or " << DOC_OPTION << "." << endl;
        return FAIL_RETURN;
    }

    char* numStr = GetCmdOption(begin, end, NUM_OPTION);
    if (!numStr) {
        cerr << "Need to specify " << NUM_OPTION << "." << endl;
        return FAIL_RETURN;
    }

    int num = atoi(numStr);
    if (num <= 0) {
        cerr << "Option " << NUM_OPTION << " should have positive value greater than 0." << endl;
        return FAIL_RETURN;
    }

    TDoc2Vec model = LoadModel(filename);
    for (const auto& word : words)
        FindAndPrintSimilarWords(model, word, num);

    for (const auto& doc : docs)
        FindAndPrintSimilarDocs(model, doc, num);

    return SUCCESS_RETURN;
}

int Vector(int argc, char* argv[]) {
    char** begin = argv + 2;
    char** end = argv + argc;

    char* filename = GetCmdOption(begin, end, LOAD_OPTION);
    if (!filename) {
        cerr << "Need to specify saved model filename with option " << LOAD_OPTION << "." << endl;
        return FAIL_RETURN;
    }

    vector<string> words = GetAllCmdOptions(begin, end, WORD_OPTION);
    vector<string> docs = GetAllCmdOptions(begin, end, DOC_OPTION);

    if (words.empty() && docs.empty()) {
        cerr << "Need to specify either " << WORD_OPTION << " or " << DOC_OPTION << "." << endl;
        return FAIL_RETURN;
    }

    TDoc2Vec model = LoadModel(filename);
    for (const auto& word : words)
        PrintWordVector(model, word);

    for (const auto& doc : docs)
        PrintDocVector(model, doc);

    return SUCCESS_RETURN;
}

void PrintHelp() {
    cout << "Doc2Vec tool" << endl
        << "There are 3 modes - 'train', 'similar', 'vector'." << endl << endl
        << "'train' mode" << endl
        << "This mode is for train doc2vec model from dataset." << endl
        << "Posible options:" << endl
        << '\t' << DATA_OPTION << " <filename> -- filename of dataset. Required option." << endl
        << '\t' << ALPHA_OPTION << " <num> -- initial learning rate. Default value: " << DEFAULT_ALPHA << '.' << endl
        << '\t' << DIMENSION_OPTION << " <num> -- dimension of word/document vectors. Default value: " << DEFAULT_DIMENSION_SIZE  << '.' << endl
        << '\t' << ITER_OPTION << " <num> -- number of iterations. Default value: " << DEFAULT_ITERATION_NUMBER << '.' << endl
        << '\t' << WINDOW_OPTION  << " <num> -- maximum skip length between words. Default value: " << DEFAULT_WINDOW_SIZE << '.' << endl
        << '\t' << THREAD_OPTION << " <num> -- number of threads. Default value: " << DEFAULT_THREAD_COUNT << '.' << endl
        << '\t' << NS_NUM_OPTION << " <num> -- number negative examples. Default value: " << DEFAULT_NEGATIVE_SAMPLE_NUMBER << '.' << endl
        << '\t' << SAMPLE_OPTION << " <num> -- threshold for occurrence of words. Popular words will be downsampled. Default value: " << DEFAULT_SAMPLE << '.' << endl
        << '\t' << HS_OPTION << " -- use Hierarchical Softmax." << endl
        << '\t' << NO_CBOW_OPTION << " -- use skip-gram model instead CBOW model." << endl
        << '\t' << SAVE_OPTION << " <filename> -- save model to file." << endl
        << endl
        << "'similar' mode" << endl
        << "This mode is for find the most similar words/docs in vocabulary/trained documents." << endl
        << "Posible options:" << endl
        << '\t' << LOAD_OPTION << " <filename> -- filename of saved model. Required option." << endl
        << '\t' << NUM_OPTION << " <num> -- number of similar words/docs to print." << endl
        << '\t' << WORD_OPTION << " <word> -- find similar words to this word." << endl
        << '\t' << DOC_OPTION << " <doc tag> -- find similar documents to document with this tag." << endl
        << endl
        << "'vector' mode" << endl
        << "This mode is for print vectors of words/docs for futher usage."
        << "Posible options:" << endl
        << '\t' << LOAD_OPTION << " <filename> -- filename of saved model. Required option." << endl
        << '\t' << WORD_OPTION << " <word> -- word to print vector." << endl
        << '\t' << DOC_OPTION << " <doc tag> -- document to print vector." << endl
        << endl
        << "EXAMPLES:" << endl
        << "Print 5 similar words from model 'model.txt' to each word." << endl
        << '\t' << "./doc2vec similar --load model.txt --num 5  --word think --word film --word queen --word strong" << endl
        << "Print vector to document." << endl
        << '\t' << "./doc2vec vector --load model.txt --doc _*2132" << endl
        << "Train model with custom parameters and save it." << endl
        << '\t' << "./doc2vec train  --save model.txt --data alldata-id.txt --hs --alpha 0.25" << endl;
};


int main(int argc, char* argv[]) {
    try {
        if (argc <= 1) {
            PrintHelp();
            return FAIL_RETURN;
        }

        if (strcmp(argv[1], HELP_OPTION.c_str()) == 0) {
            PrintHelp();
            return SUCCESS_RETURN;
        } else if (strcmp(argv[1], "train") == 0) {
            return Train(argc, argv);
        } else if (strcmp(argv[1], "similar") == 0) {
            return Similar(argc, argv);
        } else if (strcmp(argv[1], "vector") == 0) {
            return Vector(argc, argv);
        } else {
            cerr << "Unknown mode: " << argv[1] << endl;
            PrintHelp();
            return FAIL_RETURN;
        }

    } catch(runtime_error& e) {
        cerr << "An exception occured:" << endl
            << '\t' << e.what() << endl;
        return FAIL_RETURN;
    }
    return SUCCESS_RETURN;
}
