#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string.h>

#ifndef TRANSFORMERS_FULL_H
#define TRANSFORMERS_FULL_H

class SingleHeadAttention{

    private:

        int dimensions;
        int output_dimension;

        //Eigen::MatrixXd output_weigths;
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

        Eigen::MatrixXd softmax(Eigen::MatrixXd scores);

        Eigen::MatrixXd scaled_dot_product(Eigen::MatrixXd Q, Eigen::MatrixXd K, Eigen::MatrixXd V, int d_k);

        void saveMatrixBinary(const Eigen::MatrixXd& matrix, const std::string& filename);
        Eigen::MatrixXd loadMatrixBinary(const std::string& filename);

    public:
    
        SingleHeadAttention();

        SingleHeadAttention(int dimensions, int output_dimension);

        ~SingleHeadAttention();

        Eigen::MatrixXd attn_mechanism(Eigen::MatrixXd input, int d_k);

        Eigen::MatrixXd get_attention_weights();

        Eigen::MatrixXd get_Q();
        Eigen::MatrixXd get_K();
        Eigen::MatrixXd get_V();

        Eigen::MatrixXd get_scores();

        Eigen::MatrixXd get_w_q();
        Eigen::MatrixXd get_w_k();
        Eigen::MatrixXd get_w_v();

        Eigen::MatrixXd get_w_o();

        Eigen::MatrixXd get_scaled_output_raw();

        void update_w_o(Eigen::MatrixXd &d_w_o, const double learning_rate, const int t);
        void update_w_k(Eigen::MatrixXd &d_w_k, const double learning_rate, const int t);
        void update_w_q(Eigen::MatrixXd &d_w_q, const double learning_rate, const int t);
        void update_w_v(Eigen::MatrixXd &d_w_v, const double learning_rate, const int t);

        void save_weights(const std::string& folder);
        void load_weights(const std::string& folder);
    
};

class FeedForward{
    private:
        
        int input_dimensions;
        int layer1_neuron_count;
        int layer2_neuron_count;

        Eigen::MatrixXd weights1;
        Eigen::MatrixXd weights2;
        Eigen::MatrixXd hidden;
        Eigen::MatrixXd ff_input;

        Eigen::MatrixXd m_weight_1;
        Eigen::MatrixXd v_weight_1;
        Eigen::MatrixXd m_weight_2;
        Eigen::MatrixXd v_weight_2;

        void saveMatrixBinary(const Eigen::MatrixXd& matrix, const std::string& filename);
        Eigen::MatrixXd loadMatrixBinary(const std::string& filename);
    
    public:

        FeedForward();

        FeedForward(int input_dimensions, int layer1_neuron_count, int layer2_neuron_count);
        ~FeedForward();

        Eigen::MatrixXd forward(Eigen::MatrixXd input);

        void update_weight_2(Eigen::MatrixXd &d_weight_2, const double learning_rate, const int t);
        void update_weight_1(Eigen::MatrixXd &d_weight_1, const double learning_rate, const int t);

        Eigen::MatrixXd get_weights_1();
        Eigen::MatrixXd get_weights_2();
        Eigen::MatrixXd get_hidden();
        Eigen::MatrixXd get_ff_input();

        void save_weights(const std::string& folder);
        void load_weights(const std::string& folder);
};

class Linear{
    private:
        Eigen::MatrixXd weights_final;
        Eigen::MatrixXd m_weights_final;
        Eigen::MatrixXd v_weights_final;

        int input_dimension;
        int vocab_size;

        void saveMatrixBinary(const Eigen::MatrixXd& matrix, const std::string& filename);
        Eigen::MatrixXd loadMatrixBinary(const std::string& filename);

    public:
        Linear();

        Linear(int input_dimension, int vocab_size);

        ~Linear();

        Eigen::RowVectorXd linearization(Eigen::RowVectorXd input);
        void update_weigths_final(Eigen::MatrixXd &d_weight_final, int t, double learning_rate);

        Eigen::MatrixXd get_weights_final();

        void save_weights(const std::string& folder);
        void load_weights(const std::string& folder);
};

class TransformersBlock{
    private:
        int seq_len;
        int dimensions;
        int attn_output_dim;
        int vocab_size;
        int num_head;
        int d_k;
        
        SingleHeadAttention attention;
        FeedForward feed_forward;
        Linear linear_layer;

        Eigen::MatrixXd input_to_hold;
        Eigen::MatrixXd calculated_input;

        Eigen::MatrixXd m_calculated_input;
        Eigen::MatrixXd v_calculated_input;

        Eigen::VectorXd gamma1;
        Eigen::VectorXd beta1;
        Eigen::VectorXd gamma2;
        Eigen::VectorXd beta2;


        // variables for backpropagation 
        Eigen::RowVectorXd logits_b;

        Eigen::MatrixXd contextualized_input;

        Eigen::MatrixXd layer_norm(Eigen::MatrixXd input, Eigen::VectorXd gamma, Eigen::VectorXd beta);
        Eigen::VectorXd softmax(Eigen::VectorXd input);
        Eigen::MatrixXd softmax_backward_simple(const Eigen::MatrixXd& d_output, const Eigen::MatrixXd& softmax_output);
        void update_input(Eigen::MatrixXd &d_raw_input, const double learning_rate, const int t);
        Eigen::MatrixXd positional_encoding(Eigen::MatrixXd input, const int embed_dimension);

        void free_mem(Eigen::MatrixXd &m);
        void free_mem(Eigen::RowVectorXd &r);
        void free_mem(Eigen::VectorXd &v);
        
        void saveMatrixBinary(const Eigen::MatrixXd &mat, const std::string &filename);
        Eigen::MatrixXd loadMatrix(const std::string &filename);

    public:

        TransformersBlock();
        TransformersBlock(int seq_len, int dimensions, int attn_output_dim, int vocab_size);
        ~TransformersBlock();

        Eigen::Index main_logic(const Eigen::MatrixXd *input_data);
        Eigen::MatrixXd backpropagation(Eigen::VectorXd actual, const double learning_rate, const int t );
        
        void load_model(const std::string& folder);
        void save_model(const std::string& folder);



};


#endif // TRANSFORMERS_FULL_H