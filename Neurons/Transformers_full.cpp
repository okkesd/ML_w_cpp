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

            Eigen::MatrixXd scores = Q * K.transpose();
            cout << "scores are calculated" << endl;
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

            output_weigths = Eigen::MatrixXd::Ones(dimensions, output_dimension);

            cout << "single head attention is initialized" << endl;
        }

        ~SingleHeadAttention(){cout << "Single head attention is deleted" << endl;}


        // Eigen::MatrixXd
        Eigen::MatrixXd attn_mechanism(Eigen::MatrixXd input){

            Eigen::MatrixXd Q = input * w_q;
            Eigen::MatrixXd K = input * w_k;
            Eigen::MatrixXd V = input * w_v;

            cout << "first step is done. Q: " << Q.rows() << "," << Q.cols() << " - K: " << K.rows() << "," << K.cols() << " - V: " << V.rows() << "," << V.cols() << endl;

            Eigen::MatrixXd scaled_output = scaled_dot_product(Q, K, V);
            cout << "scaled_dot_product is done, scaled_output: " << scaled_output.rows() << "," << scaled_output.cols() << endl;
            
            //Eigen::MatrixXd attention_output = scaled_output.transpose() * attention_weights;
            cout << "attention_output is done"<< endl;

            return scaled_output;
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

        FeedForward(){}

        FeedForward(int input_dimensions, int layer1_neuron_count, int layer2_neuron_count) : input_dimensions(input_dimensions), 
        layer1_neuron_count(layer1_neuron_count), layer2_neuron_count(layer2_neuron_count){

            // initialize weights
            weights1 = Eigen::MatrixXd(input_dimensions, input_dimensions);
            weights2 = Eigen::MatrixXd(input_dimensions, input_dimensions);
        }

        ~FeedForward(){cout << "FeedForward is deleted" << endl;}

        Eigen::MatrixXd forward(Eigen::MatrixXd input){

            Eigen::MatrixXd out_first = input * weights1;

            Eigen::MatrixXd out_second = out_first * weights2;

            return out_second;
        }

};

class TransformersBlock{

    private:

        int dimensions;
        int attn_output_dim;
        int vocab_size;
        
        SingleHeadAttention attention;
        FeedForward feed_forward;

        Eigen::MatrixXd input;

        //Eigen::MatrixXd weights1;
        //Eigen::MatrixXd weights1;

        Eigen::MatrixXd weights_final;

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
                Eigen::RowVectorXd normalized = (row.array() -  mean) / sqrt(var + 1e-8);

                normal_input.row(i) = gamma.array() * normalized.transpose().array() + beta.array();
            }
            cout << "layer norm is done"<< endl;
            return normal_input;
        }

    public:

        TransformersBlock(){
            attention = SingleHeadAttention();
            feed_forward = FeedForward();
        }

        TransformersBlock(int dimensions, int attn_output_dim, int vocab_size): dimensions(dimensions), attn_output_dim(attn_output_dim), vocab_size(vocab_size) {

            attention = SingleHeadAttention(dimensions, attn_output_dim);
            feed_forward = FeedForward(dimensions, 4, 4);

            weights_final = Eigen::MatrixXd(dimensions, vocab_size);

            gamma1 = Eigen::VectorXd::Ones(dimensions);
            beta1 = Eigen::VectorXd::Ones(dimensions);
            gamma2 = Eigen::VectorXd::Ones(dimensions);
            beta2 = Eigen::VectorXd::Ones(dimensions);
        }

        ~TransformersBlock(){cout << "Transformers block is deleted" << endl;}

        Eigen::MatrixXd main_logic(Eigen::MatrixXd input_data){

            input = input_data;

            Eigen::MatrixXd input_normal = layer_norm(input, gamma1, beta1);
            

            Eigen::MatrixXd attn_outputs = attention.attn_mechanism(input_normal);
            cout << "attn_mechanism is done, attn_outputs: " << attn_outputs.rows() << "," << attn_outputs.cols() << endl;
            input = input + attn_outputs;

            Eigen::MatrixXd input_normal2 = layer_norm(input, gamma2, beta2);

            // give input_normal2 to the feed_forward, do the job in there, add the results to input and then return that input as contexualized input, 
            Eigen::MatrixXd ff_output = feed_forward.forward(input_normal2);
            cout << "ff is done: ff_output: " << ff_output.rows() << "," << ff_output.cols() << endl;

            input = input + ff_output;

            return input;
            // then linearization -> just one layer
        }
};

class Linear{
    private:
        Eigen::MatrixXd weights_final;
        int input_dimeansion;
        int vocab_size;

    public:
        Linear(){}

        Linear(int input_dimeansion, int vocab_size): input_dimeansion(input_dimeansion), vocab_size(vocab_size) {

            weights_final = Eigen::MatrixXd::Random(input_dimeansion, vocab_size) * 0.01;
        }

        Eigen::RowVectorXd linearization(Eigen::RowVectorXd input){

            Eigen::RowVectorXd logits = input * weights_final;

            return logits;
        }
};

Eigen::VectorXd softmax(Eigen::VectorXd input){

    Eigen::VectorXd scores(input.size());
    double max_val = input.maxCoeff();
    Eigen::VectorXd exp_scores = (input.array() - max_val).exp();
    scores = exp_scores / exp_scores.sum();

    return scores;
}

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

// to compile and run: g++ SingleHead.cpp -o SingleHead -I ../eigen-3.4.0/ && ./SingleHead

int main(){


    const int sequntial_len = 3;
    const int embedding_dimension = 5;
    const int vocab_size = 10;
    srand(time(nullptr));

    // TEST
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(sequntial_len, embedding_dimension);


    
    Eigen::MatrixXd encoded_input = input + positional_encoding(input, embedding_dimension);
    /*input<< 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15;*/

    TransformersBlock tr_block(embedding_dimension, embedding_dimension, vocab_size); // attn_output_dim is the same with embed_dim for now

    Eigen::MatrixXd contextualized_input = tr_block.main_logic(encoded_input);

    //cout << contextualized_input << endl;
    Eigen::RowVectorXd last_token = contextualized_input.row(contextualized_input.rows() -1);

    Linear linear_layer(embedding_dimension, vocab_size);

    Eigen::RowVectorXd logits = linear_layer.linearization(last_token);

    cout << "last token : " << last_token << endl;
    cout << "logits: " <<  logits << endl;

    Eigen::RowVectorXd after_softmax = softmax(logits);

    cout << "after softmax :" << after_softmax << endl;

    double predicted_token_score = after_softmax.maxCoeff();
    
    cout << "predicted token score :" << predicted_token_score << endl;

    Eigen::Index maxIndex;
    after_softmax.maxCoeff(&maxIndex);

    cout << "Predicted token index: " << maxIndex << endl;

    return 0;
}