#include <stdio.h>
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
        Eigen::MatrixXd scaled_dot_product(Eigen::MatrixXd Q, Eigen::MatrixXd K, Eigen::MatrixXd V){

            Eigen::MatrixXd scores = Q * K;

            // APPLY SOFTMAX HERE
            this->attention_weights = softmax(scores);

            Eigen::MatrixXd attention_output = attention_weights * V;
            return attention_output;
        }


    public:

        SingleHeadAttention(){}

        SingleHeadAttention(int dimensions, int output_dimension) : dimensions(dimensions), output_dimension(output_dimension) {
            
            w_q = Eigen::MatrixXd::Ones(dimensions, dimensions);
            w_k = Eigen::MatrixXd::Ones(dimensions, dimensions);
            w_v = Eigen::MatrixXd::Ones(dimensions, dimensions);

            output_weigths = Eigen::MatrixXd::Ones(dimensions, output_dimension);

            cout << "single head attention is initialized" << endl;
        }

        ~SingleHeadAttention(){cout << "Single head attention is deleted" << endl;}


        // Eigen::MatrixXd
        Eigen::MatrixXd attn_mechanism(Eigen::MatrixXd input){

            Eigen::MatrixXd Q = input * w_q;
            Eigen::MatrixXd K = input * w_k;
            Eigen::MatrixXd V = input * w_v;


            Eigen::MatrixXd scaled_output = scaled_dot_product(Q, K, V);

            Eigen::MatrixXd attention_output = scaled_output * attention_weights;

            return attention_output;
        }
};

class FeedForward{

    private:
        
        int input_dimensions;
        int layer1_neuron_count;
        int layer2_neuron_count;

        Eigen::MatrixXd weights1;
        Eigen::MatrixXd weights2;

    public:

        FeedForward(int input_dimensions, int layer1_neuron_count, int layer2_neuron_count) : input_dimensions(input_dimensions), 
        layer1_neuron_count(layer1_neuron_count), layer2_neuron_count(layer2_neuron_count){

            // initialize weights
            weights1 = Eigen::MatrixXd(input_dimensions, input_dimensions);
        }

};

class TransformersBlock{

    private:

        int dimensions;
        int attn_output_dim;
        
        SingleHeadAttention attention;

        Eigen::MatrixXd input;

        Eigen::MatrixXd weights1;
        Eigen::MatrixXd weights1;

        // for normalization
        Eigen::VectorXd gamma1;
        Eigen::VectorXd beta1;
        Eigen::VectorXd gamma2;
        Eigen::VectorXd beta2;

        Eigen::MatrixXd layer_norm(Eigen::MatrixXd input, Eigen::VectorXd gamma, Eigen::VectorXd beta){

            Eigen::MatrixXd normal_input(input.rows(), input.cols());

            for (int i = 0; i<input.rows(); i++){

                Eigen::RowVectorXd row = input.row(i);
                double mean = row.mean();
                double var = (row.array() - mean).square().mean();

                // todo: make mean a vector and substract element wise
                Eigen::VectorXd normalized = (row - Eigen::VectorXd::Ones(row.size()) * mean) / sqrt(var + 1e-8);

                return (Eigen::VectorXd::Ones(normalized.size()) * gamma) * normalized + beta;
            }
        }

    public:

        TransformersBlock(){}

        TransformersBlock(int dimensions, int attn_output_dim): attn_output_dim(attn_output_dim) {

            attention = SingleHeadAttention(dimensions, attn_output_dim);

            gamma1 = Eigen::VectorXd::Ones(dimensions);
            beta1 = Eigen::VectorXd::Ones(dimensions);
            gamma2 = Eigen::VectorXd::Ones(dimensions);
            beta2 = Eigen::VectorXd::Ones(dimensions);
        }

        ~TransformersBlock(){cout << "Transformers block is deleted" << endl;}

        void main_logic(Eigen::MatrixXd input_data){

            input = input_data;

            Eigen::MatrixXd input_normal = layer_norm(input, gamma1, beta1);
            Eigen::MatrixXd attn_outputs = attention.attn_mechanism(input_normal);
            input = input + attn_outputs;

            Eigen::MatrixXd input_normal2 = layer_norm(input, gamma2, beta2);

            // give input_normal2 to the feed_forward, do the job in there, add the results to input and then return that input as contexualized input, 
            // then linearization -> just one layer
        }
};

int main(){

    // TEST
    
    return 0;
}