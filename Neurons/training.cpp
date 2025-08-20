#include <stdio.h>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "Transformers_full.h"
#include <chrono>

using namespace std;

Eigen::MatrixXd load_embeddings(const std::string &path, size_t vocab_size, size_t dim) {
    std::ifstream file(path, std::ios::binary);
    Eigen::MatrixXd embeddings(vocab_size, dim);
    file.read(reinterpret_cast<char*>(embeddings.data()), vocab_size * dim * sizeof(double));
    return embeddings;
}

void save_embeddings(const std::string &path, const Eigen::MatrixXd &emb) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(emb.data()), emb.rows() * emb.cols() * sizeof(double));
}

vector<int32_t> load_tokens(const string &path) {
    ifstream file(path, ios::binary);
    file.seekg(0, ios::end);
    size_t size = file.tellg();
    //cout << "total size is: " << size << endl;
    file.seekg(0);
    vector<int32_t> data(size / sizeof(int32_t));
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(int32_t));
    return data;
}

using RowMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMatrixXi load_tokens_eigen(const string &path, int window_size) {
    ifstream file(path, ios::binary | ios::ate);
    if(!file) {
        cerr << "Cannot open file: " << path << endl;
        exit(1);
    }

    size_t file_size = file.tellg();
    file.seekg(0);

    size_t num_tokens = file_size / sizeof(int32_t);
    size_t num_samples = num_tokens / window_size;

    RowMatrixXi tokens(num_samples, window_size);

    file.read(reinterpret_cast<char*>(tokens.data()), num_samples * window_size * sizeof(int32_t));

    return tokens;
}


// Simple example: update embeddings (gradient descent step)
void update_embedding(Eigen::MatrixXd &embed_table, const Eigen::RowVectorXd &new_embed, int token_id) {
    embed_table.row(token_id) = new_embed;
    cout << "updated embedding " << token_id << endl;
}

void saveMatrixBinary(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out) throw std::runtime_error("Could not open file for writing: " + filename);

    Eigen::Index rows = matrix.rows();
    Eigen::Index cols = matrix.cols();

    out.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
    out.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));

    out.close();
}

Eigen::MatrixXd loadMatrixBinary(const std::string& filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in) throw std::runtime_error("Could not open file for reading: " + filename);

    Eigen::Index rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(Eigen::Index));
    in.read(reinterpret_cast<char*>(&cols), sizeof(Eigen::Index));

    Eigen::MatrixXd matrix(rows, cols);
    in.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(double));

    in.close();
    return matrix;
}


// to run and compile : g++ training.cpp Transformers_full.cpp -o training -I ../eigen-3.4.0/ && ./training

// NOTE: we need to apply backprop for layer norm too but thats for later

int main(){

    size_t vocab_size = 50281; // tiktoken o200k_base vocab size
    size_t dim = 768;
    int window_size = 3;
    const size_t batch_size = 10000; // number of samples per batch
    std::string source_folder = "weights";

    
    //Eigen::MatrixXd random_one = Eigen::MatrixXd::Ones(10,20);

    //saveMatrixBinary(random_one, source_folder + "/m_w_k.bin");

    Eigen::MatrixXd loaded_one = loadMatrixBinary(source_folder + "/weights_final.bin");

    cout << loaded_one.rows() << " - " << loaded_one.cols() << endl;


    //cout << "start : " << v.max_size() << endl;

    // Load embeddings and samples
    /*Eigen::MatrixXd embeddings = load_embeddings("./source/embedding_table.bin", vocab_size, dim);  
    RowMatrixXi inputs = load_tokens_eigen("./source/train_inputs.bin", window_size);
    vector<int32_t> targets = load_tokens("./source/train_targets.bin");

    cout << embeddings.cols() << endl;

    cout << "sample count: " << inputs.rows() << endl;
    //cout << "size: " << targets.size() << endl;
    
    //int seq_len = 3;
    int embedding_dimension = embeddings.cols(); // 768 for now
    int vocab_size_int = embeddings.rows();
    TransformersBlock tr_block(window_size, embedding_dimension, embedding_dimension, vocab_size_int);

    tr_block.load_model(source_folder);

    for (int loop = 5; loop<10; loop++){

        Eigen::MatrixXd input_data(window_size, embedding_dimension);
        for (int i = 0; i<window_size; i++){
            input_data.row(i) == embeddings.row(inputs.row(loop)[i]); // --> ids
        }
         
        auto start = chrono::high_resolution_clock::now();
    
        Eigen::Index index_returned = tr_block.main_logic(&input_data);
    
        Eigen::VectorXd actual = Eigen::VectorXd::Zero(vocab_size); // embeddings.row(targets[loop]); // static_cast<Eigen::Index>
        actual[targets[loop]] = 1;
        Eigen::MatrixXd calculated_input = tr_block.backpropagation(actual, 1, 1);
        
    
        auto end = chrono::high_resolution_clock::now();
    
        chrono::duration<double> elapsed = end - start;
    
        cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    
        auto start_embed = chrono::high_resolution_clock::now();
    
        for (int i = 0; i<window_size; i++){
            update_embedding(embeddings, calculated_input.row(i), inputs.row(loop)[i]);
        }
        auto end_embed = chrono::high_resolution_clock::now();
    
        chrono::duration<double> elapsed_embed = end_embed - start_embed;
    
        cout << "Elapsed time in updating embeddings: " << elapsed_embed.count() << " seconds\n";
    }

    tr_block.save_model(source_folder);
    return 0;*/
}

/*
67907   198  5429-->198
198 427 198-->65700
33153   198  2878-->198
198 427 198-->82
33153   198  2878-->70713
*/