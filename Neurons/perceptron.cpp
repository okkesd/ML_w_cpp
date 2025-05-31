#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <iostream>

using namespace std;

class Neuron{
    private:
        int state;
        


};

class InputLayer{
    private:
        int neuron_count;
        int next_neuron_count;
        Eigen::MatrixXd weights_to_next;
        Eigen::MatrixXd output;

    public:
        
        InputLayer(int neuron_count, int next_neuron_count) : neuron_count(neuron_count), next_neuron_count(next_neuron_count) {
            //neurons = Eigen::VectorXd(neuron_count);
            weights_to_next = Eigen::MatrixXd(neuron_count, next_neuron_count); // weights should be from neurons in input to the neurons in the next layer
            output = Eigen::MatrixXd(neuron_count, next_neuron_count); // neuron_count is rows, we will store the output of one neuron in a row of matrix
        }

        void adjust_weights(Eigen::MatrixXd weights){

            if (!weights.size() == weights_to_next.size()){
                return;
            }

            weights_to_next = weights;  
        }

        void adjust_weights(){
            for (int i = 0; i<weights_to_next.rows(); i++){
                weights_to_next.row(i) = Eigen::RowVectorXd::Ones(next_neuron_count);
            }
        }

        // we need to think about how the input data will be formed to adjust the logic 
        // (in terms of data types) ofr  passage of informations
        void main_logic(Eigen::MatrixXd input_data){  
            
            if (!input_data.cols() == neuron_count){
                return;
            }

            for (int i = 0; i<input_data.rows(); i++){ // loop for data

                Eigen::RowVectorXd one_sample = input_data.row(i); //(output.cols());

                for (int k = 0; k<one_sample.size(); k++){ // loop for the input (so that every neuron in layer will take its feature)

                    double input = one_sample[k];
                    
                    Eigen::RowVectorXd output_row(weights_to_next.row(k).size());

                    for (int j = 0; j<weights_to_next.row(k).size(); j++){ // loop for multiplying the feature with weights
                        output_row[j] = weights_to_next.row(k)[j] * input;
                    }

                    // put that feature * weights into the output. Every row of output will be fed into the next layer as data
                    output.row(k) += output_row;
                }
            }
        }
        Eigen::MatrixXd get_output(){
            return output;
        }

        Eigen::MatrixXd get_weights(){
            return weights_to_next;
        }
};

class FirstLayer{

    private:
        const int neuron_count;
        const int activation_threshold;
        vector<int> state;
        Eigen::VectorXd weights_to_next;
        Eigen::VectorXd output;
    
    public:

        FirstLayer(int neuron_count, int next_neuron_count, const int activation_threshold) : neuron_count(neuron_count), activation_threshold(activation_threshold){
            state = vector<int>(neuron_count, -1);
            weights_to_next = Eigen::VectorXd(neuron_count);
            output = Eigen::VectorXd(neuron_count);
        }

        void set_actives_with_TLU(Eigen::MatrixXd inputs){ // TLU to set neurons active or inactive based on their input

            if (!inputs.cols() == state.size()) {cout << "errror in here " << endl;return;}
            for (int i = 0; i<inputs.cols(); i++){

                double sum = inputs.col(i).sum();
                state[i] = sum >= activation_threshold ? 1 : 0;
            }
        }

        void adjust_weigths(Eigen::VectorXd weights){
            weights_to_next = weights;
        }
        void adjust_weigths(){
            weights_to_next = Eigen::VectorXd::Ones(neuron_count);
        }

        vector<int> get_state(){
            return state;
        }
        
        void main_logic(Eigen::MatrixXd input_data){

            if (!input_data.cols() == state.size()) cout << "we got problems" << endl;

            for (int i = 0; i<state.size(); i++){ // loop for neurons (and also weights_to_next since it's a perceptron)

                if (state[i] == 0) { // neuron is inactive, so asing 0 as output
                    
                } else { // neuron is active

                    //for (int k = 0; k<input_data.cols(); k++){ // get the sum of the data fed into the ith neuron

                        //cout << "i is: " << i << endl;
                        double input = input_data.col(i).sum();
                        output[i] = input * weights_to_next[i]; // assing the output as sum(the_data * the weight)
                    //}
                    
                }
            }
        }
        Eigen::VectorXd get_output(){
            return output;
        }
};


class OutputLayer{

    private:
        const int neuron_count;
        int output; // 1 or 0 if classification

        double sigmoid_function(double input) {
            return (1.0 / (1.0 + exp(-input)));
        }
    
    public:
        
        OutputLayer(const int neuron_count) : neuron_count(neuron_count) {}

        void main_logic(Eigen::VectorXd input){

            double sum = input.sum();
            double decision_sum = sigmoid_function(sum);
            
            output = decision_sum >= 0.5 ? 1 : 0;
        }

        int get_output(){
            return output;
        }
};

int main(){


    Eigen::MatrixXd data(4,4);
    Eigen::RowVectorXd row1(4);
    row1 << 5,2,4,4;
    Eigen::RowVectorXd row2(4);
    row2 << 7,4,6,7;
    Eigen::RowVectorXd row3(4);
    row3 << 2,4,3,3;
    Eigen::RowVectorXd row4(4);
    row4 << 3,0,1,2;
    
    data.row(0) = row1;
    data.row(1) = row2;
    data.row(2) = row3;
    data.row(3) = row4;

    //cout << "data\n" << data << endl;

    Eigen::VectorXd output(4);
    output << 1,1,0,0;

    //cout << output << endl;

    InputLayer input_layer(4, 5);

    input_layer.adjust_weights();

    //Eigen::MatrixXd weigths_first = input_layer.get_weights();
    //cout << weigths_first << endl;

    input_layer.main_logic(data);

    Eigen::MatrixXd out_input = input_layer.get_output();

    cout << "output of input layer\n" << out_input << endl;

    FirstLayer first_layer(5,1,34);

    first_layer.set_actives_with_TLU(out_input);

    vector<int> states = first_layer.get_state();
    
    first_layer.adjust_weigths();
    
    first_layer.main_logic(out_input);
    
    Eigen::VectorXd out_first = first_layer.get_output();

    cout << "output of first layer: " << endl;
    for (auto i : out_first){
        cout << i << endl;
    }

    OutputLayer output_layer(1);

    output_layer.main_logic(out_first);

    int output_general = output_layer.get_output();

    cout << "General output: " << output_general << endl;
    return 0;
}