#include <stdio.h>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <cassert>
#include "Transformers_full.h"
#include <fstream>
#include <iostream>
#include <string.h>

using namespace std;

/*class SingleHeadAttention{

    private:

        int dimensions;
        int output_dimension;

        Eigen::MatrixXd output_weigths;
        Eigen::MatrixXd w_q;
        Eigen::MatrixXd w_k;
        Eigen::MatrixXd w_v;
        Eigen::MatrixXd w_o;

        Eigen::MatrixXd m_w_o;
        Eigen::MatrixXd v_w_o;

        Eigen::MatrixXd m_w_k;
        Eigen::MatrixXd v_w_k;

        Eigen::MatrixXd m_w_q;
        Eigen::MatrixXd v_w_q;

        Eigen::MatrixXd m_w_v;
        Eigen::MatrixXd v_w_v;

        Eigen::MatrixXd Q;
        Eigen::MatrixXd K;
        Eigen::MatrixXd V;

        Eigen::MatrixXd attn_scores;
        Eigen::MatrixXd scaled_output_raw_b;

        Eigen::MatrixXd attention_weights;
        Eigen::MatrixXd output;
*/
        Eigen::MatrixXd SingleHeadAttention::softmax(Eigen::MatrixXd scores){

            Eigen::MatrixXd soft_scores(scores.rows(), scores.cols());
   
            for (int i = 0; i<soft_scores.rows(); i++){

                Eigen::RowVectorXd row = scores.row(i);
                double max_val = row.maxCoeff();

                Eigen::RowVectorXd exp_scores = (row.array() - max_val).exp();

                soft_scores.row(i) = exp_scores / exp_scores.sum();
            }

            return soft_scores;
        }

        // should return a matrix
        Eigen::MatrixXd SingleHeadAttention::scaled_dot_product(Eigen::MatrixXd Q, Eigen::MatrixXd K, Eigen::MatrixXd V, int d_k){

            Eigen::MatrixXd scores = Q * K.transpose() / sqrt(d_k);
            cout << "scores are calculated" << endl;
            attn_scores = scores;
            // APPLY SOFTMAX HERE
            this->attention_weights = softmax(scores);

            
            Eigen::MatrixXd attention_output = attention_weights * V;

            cout << "attn_wiehts: " << attention_weights.rows() << "," << attention_weights.cols() << endl;
            return attention_output;
        }

        void SingleHeadAttention::saveMatrixBinary(const Eigen::MatrixXd& matrix, const std::string& filename) {

            std::ofstream out(filename, std::ios::out | std::ios::binary);
            if (!out) throw std::runtime_error("Could not open file for writing: " + filename);
        
            Eigen::Index rows = matrix.rows();
            Eigen::Index cols = matrix.cols();
        
            out.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
            out.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));
        
            out.close();
        }

        Eigen::MatrixXd SingleHeadAttention::loadMatrixBinary(const std::string& filename) {
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
    

        SingleHeadAttention::SingleHeadAttention(){}

        SingleHeadAttention::SingleHeadAttention(int dimensions, int output_dimension) : dimensions(dimensions), output_dimension(output_dimension) {
            
            w_q = Eigen::MatrixXd::Ones(dimensions, dimensions);
            w_k = Eigen::MatrixXd::Ones(dimensions, dimensions);
            w_v = Eigen::MatrixXd::Ones(dimensions, dimensions);
            w_o = Eigen::MatrixXd::Ones(dimensions, dimensions);

            m_w_o = Eigen::MatrixXd::Zero(dimensions, dimensions);
            v_w_o = Eigen::MatrixXd::Zero(dimensions, dimensions);

            m_w_k = Eigen::MatrixXd::Zero(dimensions, dimensions);
            v_w_k = Eigen::MatrixXd::Zero(dimensions, dimensions);

            m_w_v = Eigen::MatrixXd::Zero(dimensions, dimensions);
            v_w_v = Eigen::MatrixXd::Zero(dimensions, dimensions);

            m_w_q = Eigen::MatrixXd::Zero(dimensions, dimensions);
            v_w_q = Eigen::MatrixXd::Zero(dimensions, dimensions);

            //output_weigths = Eigen::MatrixXd::Ones(dimensions, output_dimension);

            cout << "single head attention is initialized" << endl;
        }

        SingleHeadAttention::~SingleHeadAttention(){cout << "Single head attention is deleted" << endl;}


        // Eigen::MatrixXd
        Eigen::MatrixXd SingleHeadAttention::attn_mechanism(Eigen::MatrixXd input, int d_k){

            Q = input * w_q;
            cout << "here" << endl;
            K = input * w_k;
            V = input * w_v;

            cout << "first step is done. Q: " << Q.rows() << "," << Q.cols() << " - K: " << K.rows() << "," << K.cols() << " - V: " << V.rows() << "," << V.cols() << endl;

            Eigen::MatrixXd scaled_output_raw = scaled_dot_product(Q, K, V, d_k);
            scaled_output_raw_b = scaled_output_raw;

            Eigen::MatrixXd scaled_output = scaled_output_raw * w_o; // 3,5 * 5,5 = 3,5
            cout << "scaled_dot_product is done, scaled_output: " << scaled_output.rows() << "," << scaled_output.cols() << endl;
            
            //Eigen::MatrixXd attention_output = scaled_output.transpose() * attention_weights;
            cout << "attention_output is done"<< endl;

            return scaled_output;
        }

        Eigen::MatrixXd SingleHeadAttention::get_attention_weights(){
            return attention_weights;
        }

        Eigen::MatrixXd SingleHeadAttention::get_Q(){
            return Q;
        }
        Eigen::MatrixXd SingleHeadAttention::get_K(){
            return K;
        }
        Eigen::MatrixXd SingleHeadAttention::get_V(){
            return V;
        }

        Eigen::MatrixXd SingleHeadAttention::get_scores(){
            return attn_scores;
        }

        Eigen::MatrixXd SingleHeadAttention::get_w_q(){
            return w_q;
        }
        Eigen::MatrixXd SingleHeadAttention::get_w_k(){
            return w_k;
        }
        Eigen::MatrixXd SingleHeadAttention::get_w_v(){
            return w_v;
        }

        Eigen::MatrixXd SingleHeadAttention::get_w_o(){
            return w_o;
        }

        Eigen::MatrixXd SingleHeadAttention::get_scaled_output_raw(){
            return scaled_output_raw_b;
        }

        void SingleHeadAttention::update_w_o(Eigen::MatrixXd &d_w_o, const double learning_rate, const int t){
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_o = beta1 * m_w_o + (1 - beta1) * d_w_o;
            v_w_o = beta2 * v_w_o + (1 - beta2) * d_w_o.array().square().matrix();
            d_w_o.resize(0,0);
            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_o / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_o / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_o = w_o - learning_rate * weight_decay * w_o;
            new_w_o = ( new_w_o - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_o = new_w_o;
        }

        void SingleHeadAttention::update_w_k(Eigen::MatrixXd &d_w_k, const double learning_rate, const int t){
         
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_k = beta1 * m_w_k + (1 - beta1) * d_w_k;
            v_w_k = beta2 * v_w_k + (1 - beta2) * d_w_k.array().square().matrix();
            d_w_k.resize(0,0);
            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_k / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_k / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_k = w_k - learning_rate * weight_decay * w_k;
            new_w_k = ( new_w_k - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_k = new_w_k;
        }

        void SingleHeadAttention::update_w_q(Eigen::MatrixXd &d_w_q, const double learning_rate, const int t){
         
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_q = beta1 * m_w_q + (1 - beta1) * d_w_q;
            v_w_q = beta2 * v_w_q + (1 - beta2) * d_w_q.array().square().matrix();
            d_w_q.resize(0,0);

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_q / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_q / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_q = w_q - learning_rate * weight_decay * w_q;
            new_w_q = ( new_w_q - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_q = new_w_q;
        }

        void SingleHeadAttention::update_w_v(Eigen::MatrixXd &d_w_v, const double learning_rate, const int t){
         
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_v = beta1 * m_w_v + (1 - beta1) * d_w_v;
            v_w_v = beta2 * v_w_v + (1 - beta2) * d_w_v.array().square().matrix();
            d_w_v.resize(0,0);

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_v / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_v / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_v = w_v - learning_rate * weight_decay * w_v;
            new_w_v = ( new_w_v - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_v = new_w_v;
        }


        void SingleHeadAttention::save_weights(const string& folder){

            saveMatrixBinary(w_q, folder+"/w_q.bin");
            saveMatrixBinary(w_k, folder+"/w_k.bin");
            saveMatrixBinary(w_v, folder+"/w_v.bin");
            saveMatrixBinary(w_o, folder+"/w_o.bin");

            saveMatrixBinary(m_w_o, folder+"/m_w_o.bin");
            saveMatrixBinary(v_w_o, folder+"/v_w_o.bin");

            saveMatrixBinary(m_w_k, folder+"/m_w_k.bin");
            saveMatrixBinary(v_w_k, folder+"/v_w_k.bin");

            saveMatrixBinary(m_w_q, folder+"/m_w_q.bin");
            saveMatrixBinary(v_w_q, folder+"/v_w_q.bin");

            saveMatrixBinary(m_w_v, folder+"/m_w_v.bin");
            saveMatrixBinary(v_w_v, folder+"/v_w_v.bin");   
            cout << "saved weights in single head attention! " << endl;
        }

        void SingleHeadAttention::load_weights(const string& folder){

            w_q = loadMatrixBinary(folder+"/w_q.bin");
            w_k = loadMatrixBinary(folder+"/w_k.bin");
            w_v = loadMatrixBinary(folder+"/w_v.bin");
            w_o = loadMatrixBinary(folder+"/w_o.bin");

            m_w_o = loadMatrixBinary(folder+"/m_w_o.bin");
            v_w_o = loadMatrixBinary(folder+"/v_w_o.bin");

            m_w_k = loadMatrixBinary(folder+"/m_w_k.bin");
            v_w_k = loadMatrixBinary(folder+"/v_w_k.bin");

            m_w_q = loadMatrixBinary(folder+"/m_w_q.bin");
            v_w_q = loadMatrixBinary(folder+"/v_w_q.bin");

            m_w_v = loadMatrixBinary(folder+"/m_w_v.bin");
            v_w_v = loadMatrixBinary(folder+"/v_w_v.bin");
            cout << "loaded single head attention weights! " << endl;
        }


        
        void FeedForward::saveMatrixBinary(const Eigen::MatrixXd& matrix, const std::string& filename) {

            std::ofstream out(filename, std::ios::out | std::ios::binary);
            if (!out) throw std::runtime_error("Could not open file for writing: " + filename);
        
            Eigen::Index rows = matrix.rows();
            Eigen::Index cols = matrix.cols();
        
            out.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
            out.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));
        
            out.close();
        }

        Eigen::MatrixXd FeedForward::loadMatrixBinary(const std::string& filename) {
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


        FeedForward::FeedForward(){}

        FeedForward::FeedForward(int input_dimensions, int layer1_neuron_count, int layer2_neuron_count) : input_dimensions(input_dimensions), 
        layer1_neuron_count(layer1_neuron_count), layer2_neuron_count(layer2_neuron_count){

            // initialize weights
            weights1 = Eigen::MatrixXd::Random(input_dimensions, input_dimensions) *0.01;
            weights2 = Eigen::MatrixXd::Random(input_dimensions, input_dimensions) *0.01;

            m_weight_1 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
            v_weight_1 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
            m_weight_2 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
            v_weight_2 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
            cout << "feed forward is initialized" << endl;
        }

        FeedForward::~FeedForward(){cout << "FeedForward is deleted" << endl;}

        Eigen::MatrixXd FeedForward::forward(Eigen::MatrixXd input){

            ff_input = input;
            hidden = input * weights1;
            //cout << "hidden : " << hidden << endl;
            Eigen::MatrixXd out_second = hidden * weights2;

            return out_second;
        }

        void FeedForward::update_weight_2(Eigen::MatrixXd &d_weight_2, const double learning_rate, const int t){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;

            m_weight_2 = beta1 * m_weight_2 * (1 - beta1) * d_weight_2;
            v_weight_2 = beta2 * v_weight_2 * (1 - beta2) * d_weight_2.array().square().matrix();


            Eigen::MatrixXd new_weight_2 = weights2 - learning_rate * weight_decay * d_weight_2;
            d_weight_2.resize(0,0); // IMPORTANT -> think something for this

            Eigen::MatrixXd m_hat = beta1 * m_weight_2 / (1 - pow(beta1,t));
            Eigen::MatrixXd v_hat = beta2 * v_weight_2 / (1 - pow(beta2,t));
            
            new_weight_2 = new_weight_2 - learning_rate * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();

            weights2 = new_weight_2;
        }

        void FeedForward::update_weight_1(Eigen::MatrixXd &d_weight_1, const double learning_rate, const int t){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;

            m_weight_1 = beta1 * m_weight_1 * (1 - beta1) * d_weight_1;
            v_weight_1 = beta2 * v_weight_1 * (1 - beta2) * d_weight_1.array().square().matrix();

            Eigen::MatrixXd m_hat = beta1 * m_weight_1 / (1 - pow(beta1,t));
            Eigen::MatrixXd v_hat = beta2 * v_weight_1 / (1 - pow(beta2,t));

            Eigen::MatrixXd new_weight_2 = weights1 - learning_rate * weight_decay * d_weight_1;
            d_weight_1.resize(0,0); // IMPORTANT -> think something about this
            new_weight_2 = new_weight_2 - learning_rate * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();

            weights2 = new_weight_2;
        }

        Eigen::MatrixXd FeedForward::get_weights_1(){
            return weights1;
        }

        Eigen::MatrixXd FeedForward::get_weights_2(){
            return weights2;
        }

        Eigen::MatrixXd FeedForward::get_hidden(){
            return hidden;
        }

        Eigen::MatrixXd FeedForward::get_ff_input(){
            return ff_input;
        }

        void FeedForward::save_weights(const string& folder){

            saveMatrixBinary(weights1, folder+"/weights1.bin");
            saveMatrixBinary(weights2, folder+"/weights2.bin");
            saveMatrixBinary(m_weight_1, folder+"/m_weight_1.bin");
            saveMatrixBinary(v_weight_1, folder+"/v_weight_1.bin");
            saveMatrixBinary(m_weight_2, folder+"/m_weight_2.bin");
            saveMatrixBinary(v_weight_2, folder+"/v_weight_2.bin");
            cout << "saved weights in feed forward! " << endl;
        }


        void FeedForward::load_weights(const string& folder){

            weights1 = loadMatrixBinary(folder+"/weights1.bin");
            weights2 = loadMatrixBinary(folder+"/weights2.bin");
    
            m_weight_1 = loadMatrixBinary(folder+"/m_weight_1.bin");
            v_weight_1 = loadMatrixBinary(folder+"/v_weight_1.bin");

            m_weight_2 = loadMatrixBinary(folder+"/m_weight_2.bin");
            v_weight_2 = loadMatrixBinary(folder+"/v_weight_2.bin");
            cout << "loaded feed forward weights! " << endl;
        }



        void Linear::saveMatrixBinary(const Eigen::MatrixXd& matrix, const std::string& filename) {

            std::ofstream out(filename, std::ios::out | std::ios::binary);
            if (!out) throw std::runtime_error("Could not open file for writing: " + filename);
        
            Eigen::Index rows = matrix.rows();
            Eigen::Index cols = matrix.cols();
        
            out.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
            out.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(double));
        
            out.close();

        }       
        
        
        Eigen::MatrixXd Linear::loadMatrixBinary(const std::string& filename) {
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


        Linear::Linear(){}

        Linear::Linear(int input_dimension, int vocab_size): input_dimension(input_dimension), vocab_size(vocab_size) {

            weights_final = Eigen::MatrixXd::Random(input_dimension, vocab_size);
            m_weights_final = Eigen::MatrixXd::Zero(input_dimension, vocab_size);
            v_weights_final = Eigen::MatrixXd::Zero(input_dimension, vocab_size);
            cout << "done: linear is initialized" << endl;
        }

        Linear::~Linear(){cout << "linear is deleted" << endl;}

        Eigen::RowVectorXd Linear::linearization(Eigen::RowVectorXd input){

            Eigen::RowVectorXd logits = input * weights_final;

            return logits;
        }

        void Linear::update_weigths_final(Eigen::MatrixXd &d_weight_final, int t, double learning_rate){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_weights_final = beta1 * m_weights_final + (1 - beta1) * d_weight_final;
            v_weights_final = beta2 * v_weights_final + (1 - beta2) * d_weight_final.array().square().matrix();
            d_weight_final.resize(0,0);

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_weights_final / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_weights_final / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_weigths_final = weights_final - learning_rate * weight_decay * weights_final;
            new_weigths_final = ( new_weigths_final - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());
            
            weights_final = new_weigths_final;

            cout << "updated weights final" << endl;
        }

        Eigen::MatrixXd Linear::get_weights_final(){
            return weights_final;
        }

        void Linear::save_weights(const string& folder){

            saveMatrixBinary(weights_final, folder+"/weights_final.bin");
            saveMatrixBinary(m_weights_final, folder+"/m_weights_final.bin");
            saveMatrixBinary(v_weights_final, folder+"/v_weights_final.bin");
            cout << "saved weights in linear! " << endl;
        }

        void Linear::load_weights(const string& folder){

            weights_final = loadMatrixBinary(folder+"/weights_final.bin");
            m_weights_final = loadMatrixBinary(folder+"/m_weights_final.bin");
            v_weights_final = loadMatrixBinary(folder+"/v_weights_final.bin");
            cout << "loaded Linear weights! " << endl;
        }




        Eigen::MatrixXd TransformersBlock::layer_norm(Eigen::MatrixXd input, Eigen::VectorXd gamma, Eigen::VectorXd beta){

            Eigen::MatrixXd normal_input(input.rows(), input.cols());

            for (int i = 0; i<input.rows(); i++){

                Eigen::RowVectorXd row = input.row(i);
                double mean = row.mean();
                double var = (row.array() - mean).square().mean();

                // todo: make mean a vector and substract element wise
                Eigen::RowVectorXd normalized = (row.array() -  mean) / sqrt(var + 1e-8);

                normal_input.row(i) = gamma.array() * normalized.transpose().array() + beta.array();
            }
            cout << "layer norm is done"<< endl;
            return normal_input;
        }

        Eigen::VectorXd TransformersBlock::softmax(Eigen::VectorXd input){

            Eigen::VectorXd scores(input.size());
            double max_val = input.maxCoeff();
            Eigen::VectorXd exp_scores = (input.array() - max_val).exp();
            scores = exp_scores / exp_scores.sum();
        
            return scores;
        }

        Eigen::MatrixXd TransformersBlock::softmax_backward_simple(const Eigen::MatrixXd& d_output, const Eigen::MatrixXd& softmax_output) {
            Eigen::MatrixXd d_input = d_output.cwiseProduct(softmax_output);  // Element-wise multiply
            
            for (int i = 0; i < d_output.rows(); i++) {
                double sum = d_input.row(i).sum();
                d_input.row(i) = d_input.row(i).array() - softmax_output.row(i).array() * sum;
            }
            return d_input;
        }

        void TransformersBlock::update_input(Eigen::MatrixXd &d_raw_input, const double learning_rate, const int t){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double weight_decay = 0.001;
            double eps = 1e-8;

            m_calculated_input = beta1 * m_calculated_input + (1 - beta1) * d_raw_input;
            v_calculated_input = beta2 * v_calculated_input + (1 - beta2) * d_raw_input.array().square().matrix();
            d_raw_input.resize(0,0);
            
            Eigen::MatrixXd m_hat = beta1 * m_calculated_input / (1 - pow(beta1,t));
            Eigen::MatrixXd v_hat = beta2 * v_calculated_input / (1 - pow(beta2,t));

            Eigen::MatrixXd new_raw_input = input_to_hold - learning_rate * weight_decay * input_to_hold;
            new_raw_input = new_raw_input - learning_rate * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();

            calculated_input = new_raw_input;
        }

        void TransformersBlock::free_mem(Eigen::MatrixXd &m){
            m.resize(0,0);
        }

        void TransformersBlock::free_mem(Eigen::RowVectorXd &r){
            r.resize(Eigen::NoChange, 0);
        }

        void TransformersBlock::free_mem(Eigen::VectorXd &v){
            v.resize(0);
        }

        TransformersBlock::TransformersBlock(){
            attention = SingleHeadAttention();
            feed_forward = FeedForward();
            linear_layer = Linear();
        }

        TransformersBlock::TransformersBlock(int seq_len, int dimensions, int attn_output_dim, int vocab_size): 
                          seq_len(seq_len), dimensions(dimensions), attn_output_dim(attn_output_dim), vocab_size(vocab_size),attention(dimensions, attn_output_dim),
                          feed_forward(dimensions, 4, 4), linear_layer(dimensions, vocab_size) {

            /*attention = SingleHeadAttention(dimensions, attn_output_dim);
            feed_forward = FeedForward(dimensions, 4, 4);
            linear_layer = Linear(dimensions, vocab_size);*/

            //weights_final = Eigen::MatrixXd(dimensions, vocab_size);
            calculated_input = Eigen::MatrixXd::Zero(seq_len, dimensions);
            m_calculated_input = Eigen::MatrixXd::Zero(seq_len, dimensions);
            v_calculated_input = Eigen::MatrixXd::Zero(seq_len, dimensions);

            gamma1 = Eigen::VectorXd::Ones(dimensions);
            beta1 = Eigen::VectorXd::Ones(dimensions);
            gamma2 = Eigen::VectorXd::Ones(dimensions);
            beta2 = Eigen::VectorXd::Ones(dimensions);

            num_head = 1;
            d_k = dimensions / num_head;
            cout << "done: transformers initialized" << endl;
        }

        TransformersBlock::~TransformersBlock(){cout << "Transformers block is deleted" << endl;}

        Eigen::Index TransformersBlock::main_logic(const Eigen::MatrixXd *input_data){

            input_to_hold = *input_data;

            Eigen::MatrixXd input = *input_data;

            input = input + positional_encoding(input, dimensions);

            Eigen::MatrixXd input_normal = layer_norm(input, gamma1, beta1);
            

            Eigen::MatrixXd attn_outputs = attention.attn_mechanism(input_normal, d_k);
            cout << "attn_mechanism is done, attn_outputs: " << attn_outputs.rows() << "," << attn_outputs.cols() << endl;
            
            input = input + attn_outputs;

            Eigen::MatrixXd input_normal2 = layer_norm(input, gamma2, beta2);

            // give input_normal2 to the feed_forward, do the job in there, add the results to input and then return that input as contexualized input, 
            Eigen::MatrixXd ff_output = feed_forward.forward(input_normal2);
            cout << "ff is done: ff_output: " << ff_output.rows() << "," << ff_output.cols() << endl;

            input = input + ff_output;
            contextualized_input = input;

            Eigen::RowVectorXd last_token = input.row(input.rows() -1);
            //return input;
            // then linearization -> just one layer

            Eigen::RowVectorXd logits = linear_layer.linearization(last_token);
            logits_b = logits;

           // cout << "logits:" << logits << endl;
            Eigen::RowVectorXd after_softmax = softmax(logits);

           // cout << "after softmax :" << after_softmax << endl;

            double predicted_token_score = after_softmax.maxCoeff();
            
            cout << "predicted token score :" << predicted_token_score << endl;
        
            Eigen::Index maxIndex;
            after_softmax.maxCoeff(&maxIndex);
        
            cout << "Predicted token index: " << maxIndex << endl;

            return maxIndex;
        }

        // IMPORTANT -> consider return type maybe pointer
        Eigen::MatrixXd TransformersBlock::backpropagation(Eigen::VectorXd actual, const double learning_rate, const int t ){

            Eigen::VectorXd d_logits = logits_b.transpose() - actual;
             
            free_mem(logits_b); //logits_b.resize(Eigen::NoChange,0);
            

            Eigen::MatrixXd d_weight_final = contextualized_input.row(contextualized_input.rows() -1).transpose() * d_logits.transpose();

            //cout << "d weights final : " << d_weight_final << endl;

            Eigen::MatrixXd weights_final = linear_layer.get_weights_final();


            Eigen::VectorXd d_cont_inp_last = (weights_final * d_logits).transpose();
            
            free_mem(weights_final); //weights_final.resize(0,0);
            free_mem(d_logits); //d_logits.resize(0);

            //cout << "d cont inp : " << d_cont_inp_last << endl;

            Eigen::MatrixXd d_cont_inp = Eigen::MatrixXd::Zero(contextualized_input.rows(), contextualized_input.cols());
            free_mem(contextualized_input);

            d_cont_inp.row(d_cont_inp.rows() -1) = d_cont_inp_last;
            free_mem(d_cont_inp_last);

            //cout << "d cont inp : " << d_cont_inp << endl;

            // feed forward starts here
            Eigen::MatrixXd d_input_ff = d_cont_inp;
            Eigen::MatrixXd d_output_ff = d_cont_inp;
            free_mem(d_cont_inp);
            

            Eigen::MatrixXd weights_2 = feed_forward.get_weights_2();
            Eigen::MatrixXd d_hidden = d_output_ff * weights_2.transpose();
            free_mem(weights_2);

            Eigen::MatrixXd hidden = feed_forward.get_hidden();
            Eigen::MatrixXd d_weight_2 = hidden.transpose() * d_output_ff;
            free_mem(d_output_ff);
            free_mem(hidden);

            //cout << "d_weight_2: " << d_weight_2 << endl;
            //cout << "d hidden: " << d_hidden << endl;

            Eigen::MatrixXd ff_input = feed_forward.get_ff_input();
            Eigen::MatrixXd d_weight_1 = ff_input.transpose() * d_hidden;
            free_mem(ff_input);

            Eigen::MatrixXd weights_1 = feed_forward.get_weights_1();
            Eigen::MatrixXd d_ff_input = d_hidden * weights_1.transpose();
            free_mem(weights_1);
            free_mem(d_hidden);
            // feed forward ends here

            //cout << "d_weight_1: " << d_weight_1 << endl;
            //cout << "d ff input: " << d_ff_input << endl;

            
            Eigen::MatrixXd d_input_after_attention = d_input_ff + d_ff_input;
            free_mem(d_ff_input);
            free_mem(d_input_ff);

            // Attention Mechanism starts here
            Eigen::MatrixXd d_input_attn = d_input_after_attention;
            Eigen::MatrixXd d_output_attn = d_input_after_attention;
            free_mem(d_input_after_attention);

            Eigen::MatrixXd w_o = attention.get_w_o();
            Eigen::MatrixXd scaled_output_raw = attention.get_scaled_output_raw();

            Eigen::MatrixXd d_w_o = scaled_output_raw.transpose() * d_output_attn;
            Eigen::MatrixXd d_scaled_output_raw = d_output_attn * w_o.transpose();
            free_mem(scaled_output_raw);
            free_mem(w_o);
            free_mem(d_output_attn);


            Eigen::MatrixXd attn_weights = attention.get_attention_weights();
            Eigen::MatrixXd d_attn_V = attn_weights.transpose() * d_scaled_output_raw; // for attn_output = attn_weights * V

            Eigen::MatrixXd attn_V = attention.get_V();
            Eigen::MatrixXd d_attn_weights = d_scaled_output_raw * attn_V.transpose();
            free_mem(d_scaled_output_raw);
            free_mem(scaled_output_raw);
            free_mem(attn_V);


            Eigen::MatrixXd d_scores = softmax_backward_simple(d_attn_weights, attn_weights);
            free_mem(d_attn_weights);
            free_mem(attn_weights);

            //Eigen::MatrixXd scores = attention.get_scores();
            Eigen::MatrixXd attn_Q = attention.get_Q();
            Eigen::MatrixXd attn_K = attention.get_K();

            Eigen::MatrixXd d_attn_Q = ( d_scores * attn_K ) / sqrt(d_k);
            Eigen::MatrixXd d_attn_K = ((attn_Q.transpose() * d_scores).transpose()) / sqrt(d_k);
            free_mem(attn_Q);
            free_mem(attn_K);
            free_mem(d_scores);

            // recreate the input with position here
            Eigen::MatrixXd input = input_to_hold + positional_encoding(input_to_hold, dimensions);

            Eigen::MatrixXd d_w_k = input.transpose() * d_attn_K;
            Eigen::MatrixXd d_w_q = input.transpose() * d_attn_Q;
            Eigen::MatrixXd d_w_v = input.transpose() * d_attn_V;

            //cout << "d_K: " << d_attn_K << endl;

            Eigen::MatrixXd w_q = attention.get_w_q();
            Eigen::MatrixXd w_k = attention.get_w_k();
            Eigen::MatrixXd w_v = attention.get_w_v();

            // every row of d_raw_input points to the gradient of the embedding
            Eigen::MatrixXd d_raw_input = d_input_attn + (d_attn_K * w_k + d_attn_Q * w_q + d_attn_V * w_v);
            free_mem(w_q);
            free_mem(w_k);
            free_mem(w_v);
            free_mem(d_attn_K);
            free_mem(d_attn_Q);
            free_mem(d_input_attn);
            free_mem(d_attn_V);
            

            // Updating
            linear_layer.update_weigths_final(d_weight_final, t, learning_rate);
            
            feed_forward.update_weight_2(d_weight_2, learning_rate, t);
            feed_forward.update_weight_1(d_weight_1, learning_rate, t);
            
            attention.update_w_o(d_w_o, learning_rate, t);

            attention.update_w_k(d_w_k, learning_rate, t);
            attention.update_w_q(d_w_q, learning_rate, t);
            attention.update_w_v(d_w_v, learning_rate, t);
            cout << "here so far so good" << endl;
            update_input(d_raw_input, learning_rate, t);
            return calculated_input;
        }

void TransformersBlock::saveMatrixBinary(const Eigen::MatrixXd &mat, const std::string &filename) {
    ofstream out_file(filename, std::ios::binary);
    int rows = mat.rows(), cols = mat.cols();
    out_file.write((char*)(&rows), sizeof(int));
    out_file.write((char*)(&cols), sizeof(int));
    out_file.write((char*)mat.data(), sizeof(double) * rows * cols);
    out_file.close();
}


Eigen::MatrixXd TransformersBlock::loadMatrix(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    int rows, cols;
    in >> rows >> cols;
    Eigen::MatrixXd mat(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in >> mat(r, c);
        }
    }
    return mat;
}

void TransformersBlock::save_model(const std::string& folder){

    /*saveMatrixBinary(gamma1, folder+"/gamma1.bin");
    saveMatrixBinary(gamma2, folder+"/gamma2.bin");
    saveMatrixBinary(beta1, folder+"/beta1.bin");
    saveMatrixBinary(beta2, folder+"/beta2.bin");*/

    attention.save_weights(folder);
    feed_forward.save_weights(folder);
    linear_layer.save_weights(folder);

    cout << "whole model saved to " << folder << endl;
}

void TransformersBlock::load_model(const string& folder){
    
    /*gamma1 = loadMatrix(folder+"/gamma1.bin");
    gamma2 = loadMatrix(folder+"/gamma2.bin");

    cout << "first phase is done " << endl;

    beta1 = loadMatrix(folder+"/beta1.bin");
    beta2 = loadMatrix(folder+"/beta2.bin");*/

    attention.load_weights(folder);
    feed_forward.load_weights(folder);
    linear_layer.load_weights(folder);

    cout << "whole model loaded from " << folder << endl;
}
        

Eigen::MatrixXd TransformersBlock::positional_encoding(Eigen::MatrixXd input, const int embed_dimension){

    Eigen::MatrixXd pos_en(input.rows(), input.cols());
    for (int i = 0; i<input.rows(); i++){

        Eigen::RowVectorXd row = input.row(i);

        for (int k = 0; k<row.size(); k++){
        
            if (k % 2 == 0){

                pos_en(i,k) = sin(i / pow(10000, 2.0*k/embed_dimension));

            } else {

                pos_en(i,k) = cos(i / pow(10000, 2.0*k/embed_dimension));
            }
        }
    }
    return pos_en;
}

// to compile and run: g++ Transformers_full.cpp -o Transformers_full -I ../eigen-3.4.0/ && ./Transformers_full

// comment out main to use it in training.cpp file

/*int main(){


    const int sequntial_len = 3;
    const int embedding_dimension = 5;
    const int vocab_size = 5;
    srand(time(nullptr));

    // TEST
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(sequntial_len, embedding_dimension);


    
    Eigen::MatrixXd encoded_input = input + positional_encoding(input, embedding_dimension);
    /*input<< 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15;*

    TransformersBlock tr_block(sequntial_len, embedding_dimension, embedding_dimension, vocab_size); // attn_output_dim is the same with embed_dim for now

    Eigen::Index predicted_token_index = tr_block.main_logic(encoded_input);
    

    Eigen::VectorXd actual(5);
    actual << 0,0,1,0,0;

    cout << "actual: " << actual << endl;
    double learning_rate = 0.001;
    int t = 1;
    tr_block.backpropagation(actual, learning_rate, t);

    return 0;
}*/