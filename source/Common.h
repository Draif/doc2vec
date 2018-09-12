#pragma once
#include <string>

const int SUCCESS_RETURN = 0;
const int FAIL_RETURN = 1;

const int MAX_EXP = 6;
const int EXP_TABLE_SIZE = 1000;
const unsigned int NEGATIVE_SAMPLE_TABLE_SIZE = 1e8;
const long long UPDATE_WORD_NUMBER = 10e4;
const double ALPHA_MAX_REDUCE_COEFFICENT = 0.0001;
const unsigned int MAX_CODE_LENGTH = 40;

const char SERIALIZE_DELIM = ' ';

const std::string WORD_OPTION = "--word";
const std::string DOC_OPTION = "--doc";
const std::string LOAD_OPTION = "--load";
const std::string SAVE_OPTION = "--save";
const std::string NUM_OPTION = "--num";
const std::string DATA_OPTION = "--data";
const std::string DIMENSION_OPTION = "--dimension";
const std::string HS_OPTION = "--hs";
const std::string NO_CBOW_OPTION = "--no-cbow";
const std::string NS_NUM_OPTION = "--ns-num";
const std::string ITER_OPTION = "--iter";
const std::string WINDOW_OPTION = "--window";
const std::string SAMPLE_OPTION = "--sample";
const std::string THREAD_OPTION = "--thread";
const std::string ALPHA_OPTION = "--alpha";
const std::string HELP_OPTION = "--help";

const unsigned int DEFAULT_DIMENSION_SIZE = 100;
const bool DEFAULT_HIERARCHICAL_SOFTMAX = false;
const bool DEFAULT_CBOW = true;
const unsigned int DEFAULT_NEGATIVE_SAMPLE_NUMBER = 5;
const unsigned int DEFAULT_ITERATION_NUMBER = 5;
const unsigned int DEFAULT_WINDOW_SIZE = 5;
const double DEFAULT_SAMPLE = 1e-3;
const unsigned int DEFAULT_THREAD_COUNT = 4;
const double DEFAULT_ALPHA = 0.05;

