#include <stdio.h>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <cassert>

using namespace std;

class SingleHeadAttention{

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

        Eigen::MatrixXd softmax(Eigen::MatrixXd scores){

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
        Eigen::MatrixXd scaled_dot_product(Eigen::MatrixXd Q, Eigen::MatrixXd K, Eigen::MatrixXd V, int d_k){

            Eigen::MatrixXd scores = Q * K.transpose() / sqrt(d_k);
            cout << "scores are calculated" << endl;
            attn_scores = scores;
            // APPLY SOFTMAX HERE
            this->attention_weights = softmax(scores);

            
            Eigen::MatrixXd attention_output = attention_weights * V;

            cout << "attn_wiehts: " << attention_weights.rows() << "," << attention_weights.cols() << endl;
            return attention_output;
        }


    public:

        SingleHeadAttention(){}

        SingleHeadAttention(int dimensions, int output_dimension) : dimensions(dimensions), output_dimension(output_dimension) {
            
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

            output_weigths = Eigen::MatrixXd::Ones(dimensions, output_dimension);

            cout << "single head attention is initialized" << endl;
        }

        ~SingleHeadAttention(){cout << "Single head attention is deleted" << endl;}


        // Eigen::MatrixXd
        Eigen::MatrixXd attn_mechanism(Eigen::MatrixXd input, int d_k){

            Q = input * w_q;
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

        Eigen::MatrixXd get_attention_weights(){
            return attention_weights;
        }

        Eigen::MatrixXd get_Q(){
            return Q;
        }
        Eigen::MatrixXd get_K(){
            return K;
        }
        Eigen::MatrixXd get_V(){
            return V;
        }

        Eigen::MatrixXd get_scores(){
            return attn_scores;
        }

        Eigen::MatrixXd get_w_q(){
            return w_q;
        }
        Eigen::MatrixXd get_w_k(){
            return w_k;
        }
        Eigen::MatrixXd get_w_v(){
            return w_v;
        }

        Eigen::MatrixXd get_w_o(){
            return w_o;
        }

        Eigen::MatrixXd get_scaled_output_raw(){
            return scaled_output_raw_b;
        }

        void update_w_o(Eigen::MatrixXd d_w_o, const double learning_rate, const int t){
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_o = beta1 * m_w_o + (1 - beta1) * d_w_o;
            v_w_o = beta2 * v_w_o + (1 - beta2) * d_w_o.array().square().matrix();

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_o / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_o / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_o = w_o - learning_rate * weight_decay * w_o;
            new_w_o = ( new_w_o - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_o = new_w_o;
        }

        void update_w_k(Eigen::MatrixXd d_w_k, const double learning_rate, const int t){
         
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_k = beta1 * m_w_k + (1 - beta1) * d_w_k;
            v_w_k = beta2 * v_w_k + (1 - beta2) * d_w_k.array().square().matrix();

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_k / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_k / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_k = w_k - learning_rate * weight_decay * w_k;
            new_w_k = ( new_w_k - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_k = new_w_k;
        }

        void update_w_q(Eigen::MatrixXd d_w_q, const double learning_rate, const int t){
         
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_q = beta1 * m_w_q + (1 - beta1) * d_w_q;
            v_w_q = beta2 * v_w_q + (1 - beta2) * d_w_q.array().square().matrix();

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_q / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_q / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_q = w_q - learning_rate * weight_decay * w_q;
            new_w_q = ( new_w_q - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_q = new_w_q;
        }

        void update_w_v(Eigen::MatrixXd d_w_v, const double learning_rate, const int t){
         
            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_w_v = beta1 * m_w_v + (1 - beta1) * d_w_v;
            v_w_v = beta2 * v_w_v + (1 - beta2) * d_w_v.array().square().matrix();

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_w_v / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_w_v / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_w_v = w_v - learning_rate * weight_decay * w_v;
            new_w_v = ( new_w_v - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());

            w_v = new_w_v;
        }
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
    
    public:

        FeedForward(){}

        FeedForward(int input_dimensions, int layer1_neuron_count, int layer2_neuron_count) : input_dimensions(input_dimensions), 
        layer1_neuron_count(layer1_neuron_count), layer2_neuron_count(layer2_neuron_count){

            // initialize weights
            weights1 = Eigen::MatrixXd::Random(input_dimensions, input_dimensions) *0.01;
            weights2 = Eigen::MatrixXd::Random(input_dimensions, input_dimensions) *0.01;

            m_weight_1 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
            v_weight_1 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
            m_weight_2 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
            v_weight_2 = Eigen::MatrixXd::Zero(input_dimensions, input_dimensions);
        }

        ~FeedForward(){cout << "FeedForward is deleted" << endl;}

        Eigen::MatrixXd forward(Eigen::MatrixXd input){

            ff_input = input;
            hidden = input * weights1;
            cout << "hidden : " << hidden << endl;
            Eigen::MatrixXd out_second = hidden * weights2;

            return out_second;
        }

        void update_weight_2(Eigen::MatrixXd d_weight_2, const double learning_rate, const int t){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;

            m_weight_2 = beta1 * m_weight_2 * (1 - beta1) * d_weight_2;
            v_weight_2 = beta2 * v_weight_2 * (1 - beta2) * d_weight_2.array().square().matrix();

            Eigen::MatrixXd m_hat = beta1 * m_weight_2 / (1 - pow(beta1,t));
            Eigen::MatrixXd v_hat = beta2 * v_weight_2 / (1 - pow(beta2,t));

            Eigen::MatrixXd new_weight_2 = weights2 - learning_rate * weight_decay * d_weight_2;
            new_weight_2 = new_weight_2 - learning_rate * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();

            weights2 = new_weight_2;
        }

        void update_weight_1(Eigen::MatrixXd d_weight_1, const double learning_rate, const int t){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;

            m_weight_1 = beta1 * m_weight_1 * (1 - beta1) * d_weight_1;
            v_weight_1 = beta2 * v_weight_1 * (1 - beta2) * d_weight_1.array().square().matrix();

            Eigen::MatrixXd m_hat = beta1 * m_weight_1 / (1 - pow(beta1,t));
            Eigen::MatrixXd v_hat = beta2 * v_weight_1 / (1 - pow(beta2,t));

            Eigen::MatrixXd new_weight_2 = weights1 - learning_rate * weight_decay * d_weight_1;
            new_weight_2 = new_weight_2 - learning_rate * (m_hat.array() / (v_hat.array().sqrt() + eps)).matrix();

            weights2 = new_weight_2;
        }

        Eigen::MatrixXd get_weights_1(){
            return weights1;
        }

        Eigen::MatrixXd get_weights_2(){
            return weights2;
        }

        Eigen::MatrixXd get_hidden(){
            return hidden;
        }

        Eigen::MatrixXd get_ff_input(){
            return ff_input;
        }

};

class Linear{
    private:
        Eigen::MatrixXd weights_final;
        Eigen::MatrixXd m_weights_final;
        Eigen::MatrixXd v_weights_final;

        int input_dimeansion;
        int vocab_size;

    public:
        Linear(){}

        Linear(int input_dimeansion, int vocab_size): input_dimeansion(input_dimeansion), vocab_size(vocab_size) {

            weights_final = Eigen::MatrixXd::Random(input_dimeansion, vocab_size) * 0.01;
            m_weights_final = Eigen::MatrixXd::Zero(input_dimeansion, vocab_size);
            v_weights_final = Eigen::MatrixXd::Zero(input_dimeansion, vocab_size);
        }

        Eigen::RowVectorXd linearization(Eigen::RowVectorXd input){

            Eigen::RowVectorXd logits = input * weights_final;

            return logits;
        }

        void update_weigths_final(Eigen::MatrixXd d_weight_final, int t, double learning_rate){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-8;
            double weight_decay = 0.01;
            
            m_weights_final = beta1 * m_weights_final + (1 - beta1) * d_weight_final;
            v_weights_final = beta2 * v_weights_final + (1 - beta2) * d_weight_final.array().square().matrix();

            // calculate m_hat and v_hat
            Eigen::MatrixXd m_hat = m_weights_final / (1 - pow(beta1, t));
            Eigen::MatrixXd v_hat = v_weights_final / (1 - pow(beta2, t));

            // I'm not sure of this, we need to clarify it
            Eigen::MatrixXd new_weigths_final = weights_final - learning_rate * weight_decay * weights_final;
            new_weigths_final = ( new_weigths_final - learning_rate * ( m_hat.array() / (v_hat.array().sqrt() + eps) ).matrix());
            
            weights_final = new_weigths_final;

            cout << "updated weights final : " << endl;
        }

        Eigen::MatrixXd get_weights_final(){
            return weights_final;
        }
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

        Eigen::MatrixXd input;
        Eigen::MatrixXd calculated_input;

        Eigen::MatrixXd m_calculated_input;
        Eigen::MatrixXd v_calculated_input;

        
        //Eigen::MatrixXd weights1;
        //Eigen::MatrixXd weights1;

        //Eigen::MatrixXd weights_final;
        

        // for normalization
        Eigen::VectorXd gamma1;
        Eigen::VectorXd beta1;
        Eigen::VectorXd gamma2;
        Eigen::VectorXd beta2;


        // variables for backpropagation 
        Eigen::RowVectorXd logits_b;

        Eigen::MatrixXd contextualized_input;

        Eigen::MatrixXd layer_norm(Eigen::MatrixXd input, Eigen::VectorXd gamma, Eigen::VectorXd beta){

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

        Eigen::VectorXd softmax(Eigen::VectorXd input){

            Eigen::VectorXd scores(input.size());
            double max_val = input.maxCoeff();
            Eigen::VectorXd exp_scores = (input.array() - max_val).exp();
            scores = exp_scores / exp_scores.sum();
        
            return scores;
        }

        Eigen::MatrixXd softmax_backward_simple(const Eigen::MatrixXd& d_output, const Eigen::MatrixXd& softmax_output) {
            Eigen::MatrixXd d_input = d_output.cwiseProduct(softmax_output);  // Element-wise multiply
            
            for (int i = 0; i < d_output.rows(); i++) {
                double sum = d_input.row(i).sum();
                d_input.row(i) = d_input.row(i).array() - softmax_output.row(i).array() * sum;
            }
            return d_input;
        }

        void update_input(Eigen::MatrixXd d_raw_input, const double learning_rate, const int t){

            double beta1 = 0.9;
            double beta2 = 0.999;
            double weight_decay = 0.001;
            double eps = 1e-8;

            m_calculated_input = beta1 * m_calculated_input + (1 - beta1) * d_raw_input;
            v_calculated_input = beta2 * v_calculated_input + (1 - beta2) * d_raw_input.array().square().matrix();
            
            Eigen::MatrixXd m_hat = beta1 * m_calculated_input / (1 - pow(beta1,t));
            Eigen::MatrixXd v_hat = beta2 * v_calculated_input / (1 - pow(beta2,t));

            Eigen::MatrixXd new_raw_input = input - learning_rate * weight_decay * input;
            new_raw_input = new_raw_input - learning_rate * (m_calculated_input.array() / (v_calculated_input.array().sqrt() + eps)).matrix();

            calculated_input = new_raw_input;
        }

    public:

        TransformersBlock(){
            attention = SingleHeadAttention();
            feed_forward = FeedForward();
            linear_layer = Linear();
        }

        TransformersBlock(int seq_len, int dimensions, int attn_output_dim, int vocab_size): 
                          seq_len(seq_len), dimensions(dimensions), attn_output_dim(attn_output_dim), vocab_size(vocab_size) {

            attention = SingleHeadAttention(dimensions, attn_output_dim);
            feed_forward = FeedForward(dimensions, 4, 4);
            linear_layer = Linear(dimensions, vocab_size);

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
        }

        ~TransformersBlock(){cout << "Transformers block is deleted" << endl;}

        Eigen::Index main_logic(Eigen::MatrixXd input_data){

            input = input_data;

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

            cout << "logits:" << logits << endl;
            Eigen::RowVectorXd after_softmax = softmax(logits);

            cout << "after softmax :" << after_softmax << endl;

            double predicted_token_score = after_softmax.maxCoeff();
            
            cout << "predicted token score :" << predicted_token_score << endl;
        
            Eigen::Index maxIndex;
            after_softmax.maxCoeff(&maxIndex);
        
            cout << "Predicted token index: " << maxIndex << endl;

            return maxIndex;
        }

        void backpropagation(Eigen::VectorXd actual, const double learning_rate, const int t ){
            
            Eigen::VectorXd d_logits = logits_b.transpose() - actual;

            cout << "d_logits : " << d_logits << endl;

            //cout << "contextualized_input last row size: " << contextualized_input.row(contextualized_input.rows() -1).size() << endl;

            //cout << "d_logits last row size: " << d_logits.size() << endl;

            Eigen::MatrixXd d_weight_final = contextualized_input.row(contextualized_input.rows() -1).transpose() * d_logits.transpose();

            cout << "d weights final : " << d_weight_final << endl;

            Eigen::MatrixXd weights_final = linear_layer.get_weights_final();

            Eigen::VectorXd d_cont_inp_last = weights_final.transpose() * d_logits;

            cout << "d cont inp : " << d_cont_inp_last << endl;

            Eigen::MatrixXd d_cont_inp = Eigen::MatrixXd::Zero(contextualized_input.rows(), contextualized_input.cols());

            d_cont_inp.row(d_cont_inp.rows() -1) = d_cont_inp_last;

            cout << "d cont inp : " << d_cont_inp << endl;

            // Feed forward
            Eigen::MatrixXd d_input_ff = d_cont_inp;
            Eigen::MatrixXd d_output_ff = d_cont_inp;

            Eigen::MatrixXd weights_2 = feed_forward.get_weights_2();
            Eigen::MatrixXd d_hidden = d_output_ff * weights_2.transpose();

            Eigen::MatrixXd hidden = feed_forward.get_hidden();
            Eigen::MatrixXd d_weight_2 = hidden.transpose() * d_output_ff;

            cout << "d_weight_2: " << d_weight_2 << endl;
            cout << "d hidden: " << d_hidden << endl;

            Eigen::MatrixXd ff_input = feed_forward.get_ff_input();
            Eigen::MatrixXd d_weight_1 = ff_input.transpose() * d_hidden;

            Eigen::MatrixXd weights_1 = feed_forward.get_weights_1();
            Eigen::MatrixXd d_ff_input = d_hidden * weights_1.transpose();

            cout << "d_weight_1: " << d_weight_1 << endl;
            cout << "d ff input: " << d_ff_input << endl;

            Eigen::MatrixXd d_input_after_attention = d_input_ff + d_ff_input;

            // Attention Mechanism
            Eigen::MatrixXd d_input_attn = d_input_after_attention;
            Eigen::MatrixXd d_output_attn = d_input_after_attention;

            Eigen::MatrixXd w_o = attention.get_w_o();
            Eigen::MatrixXd scaled_output_raw = attention.get_scaled_output_raw();

            Eigen::MatrixXd d_w_o = scaled_output_raw.transpose() * d_output_attn;
            Eigen::MatrixXd d_scaled_output_raw = d_output_attn * w_o.transpose();


            Eigen::MatrixXd attn_weights = attention.get_attention_weights();
            Eigen::MatrixXd d_attn_V = attn_weights.transpose() * d_scaled_output_raw; // for attn_output = attn_weights * V

            Eigen::MatrixXd attn_V = attention.get_V();
            Eigen::MatrixXd d_attn_weights = d_scaled_output_raw * attn_V.transpose();


            Eigen::MatrixXd d_scores = softmax_backward_simple(d_attn_weights, attn_weights);

            Eigen::MatrixXd scores = attention.get_scores();
            Eigen::MatrixXd attn_Q = attention.get_Q();
            Eigen::MatrixXd attn_K = attention.get_K();

            Eigen::MatrixXd d_attn_Q = ( d_scores * attn_K ) / sqrt(d_k);
            Eigen::MatrixXd d_attn_K = ((attn_Q.transpose() * d_scores).transpose()) / sqrt(d_k);

            Eigen::MatrixXd d_w_k = input.transpose() * d_attn_K;
            Eigen::MatrixXd d_w_q = input.transpose() * d_attn_Q;
            Eigen::MatrixXd d_w_v = input.transpose() * d_attn_V;

            cout << "d_K: " << d_attn_K << endl;

            Eigen::MatrixXd w_q = attention.get_w_q();
            Eigen::MatrixXd w_k = attention.get_w_k();
            Eigen::MatrixXd w_v = attention.get_w_v();

            // every row of d_raw_input points to the gradient of the embedding
            Eigen::MatrixXd d_raw_input = d_input_attn + (d_attn_K * w_k + d_attn_Q * w_q + d_attn_V * w_v);
            

            // Updating
            linear_layer.update_weigths_final(d_weight_final, t, learning_rate);
            
            feed_forward.update_weight_2(d_weight_2, learning_rate, t);
            feed_forward.update_weight_1(d_weight_1, learning_rate, t);
            
            attention.update_w_o(d_w_o, learning_rate, t);
            
            attention.update_w_k(d_w_k, learning_rate, t);
            attention.update_w_q(d_w_q, learning_rate, t);
            attention.update_w_v(d_w_v, learning_rate, t);
            cout << "here so far" << endl;
            update_input(d_raw_input, learning_rate, t);
        }

};



Eigen::MatrixXd positional_encoding(Eigen::MatrixXd input, const int embed_dimension){

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

int main(){


    const int sequntial_len = 3;
    const int embedding_dimension = 5;
    const int vocab_size = 5;
    srand(time(nullptr));

    // TEST
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(sequntial_len, embedding_dimension);


    
    Eigen::MatrixXd encoded_input = input + positional_encoding(input, embedding_dimension);
    /*input<< 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15;*/

    TransformersBlock tr_block(sequntial_len, embedding_dimension, embedding_dimension, vocab_size); // attn_output_dim is the same with embed_dim for now

    Eigen::Index predicted_token_index = tr_block.main_logic(encoded_input);
    

    Eigen::VectorXd actual(5);
    actual << 0,0,1,0,0;

    cout << "actual: " << actual << endl;
    double learning_rate = 0.001;
    int t = 1;
    tr_block.backpropagation(actual, learning_rate, t);

    return 0;
}