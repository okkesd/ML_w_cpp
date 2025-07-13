#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <cassert>

using namespace std;

class Neuron{
    private:
        int state;
};

class Layer{

    private:

        Eigen::MatrixXd weights_to_next;

    public:

        Layer(){}

        Eigen::MatrixXd get_weights(){
            return weights_to_next;
        }
};

class InputLayer{
    private:
        int neuron_count;
        int next_neuron_count;
        Eigen::VectorXd input;
        Eigen::MatrixXd weights_to_next; // the first row represents weights from firt neuron to the next layer
                                         // the firts column represents weights to the first neuron in the next layer

        Eigen::VectorXd output;

    public:
        
        // constructor
        InputLayer(int neuron_count, int next_neuron_count) : neuron_count(neuron_count), next_neuron_count(next_neuron_count) {

            weights_to_next = Eigen::MatrixXd(neuron_count, next_neuron_count); // weights should be from neurons in input to the neurons in the next layer
            output = Eigen::VectorXd(next_neuron_count); // neuron_count is rows, we will store the output of one neuron in a row of matrix
        }

        // destructor
        ~InputLayer(){}

        void adjust_weights(Eigen::RowVectorXd weights){

            if (!weights.size() == weights_to_next.size()){
                return;
            }

            weights_to_next = weights;  
            cout << "weights adjusted" << endl;
        }

        void adjust_weights(){
            
            weights_to_next = Eigen::MatrixXd::Ones(neuron_count, next_neuron_count);
            
        }

        void update_one_weight(int column_location, int row_location, double value){
            weights_to_next(row_location, column_location) = value;
        }

        void main_logic(Eigen::VectorXd input_data){  
            /* main logic to process input via predefined weights */
            
            if (input_data.size() != neuron_count){
                return;
            }

            input = input_data;

            // calcualte the output
            for (int i = 0; i<next_neuron_count; i++){
                
                double sum = 0;
                for (int k = 0; k<input_data.size(); k++){
                    sum += input_data[k] * weights_to_next.col(i)[k];
                }
                output[i] = sum;
            }
                
        }
        Eigen::VectorXd get_output(){
            return output;
        }

        Eigen::MatrixXd get_weights(){
            return weights_to_next;
        }
        void print_weights(){
            
            cout << "Weights: " << weights_to_next << endl;
        }

        int get_neuron_count(){
            return neuron_count;
        }
};

class HiddenLayer{

    private:
        const int neuron_count;
        const int next_neuron_count;
        
        Eigen::VectorXd input;
        Eigen::MatrixXd weights_to_next; // the first row represents the weights from the first neuron to the next layer
                                         // the first column represents the weights to the first neuron in the next layer

        Eigen::VectorXd output;          // the first element represents the input to the next layer's first neuron

        Eigen::VectorXd neuron_outputs;  // the first element represents the output of the activation function of the first neuron

        vector<double> ds;               // to hold the gradients (ds) of layer

        double sigmoid(double input){
            return (1.0 / (1.0 + exp(-input)));
        }
    
    public:

        HiddenLayer(const int neuron_count, const int next_neuron_count) : neuron_count(neuron_count),
        next_neuron_count(next_neuron_count), ds(neuron_count, 0) {
        
            weights_to_next = Eigen::MatrixXd(neuron_count, next_neuron_count);
            output = Eigen::VectorXd(next_neuron_count);
            neuron_outputs = Eigen::VectorXd::Zero(neuron_count);
        }

        void adjust_weigths(Eigen::VectorXd weights){
            weights_to_next = weights;
        }

        void adjust_weigths(){
            weights_to_next = Eigen::MatrixXd::Ones(neuron_count, next_neuron_count);
        }

        void update_one_weight(int column_location, int row_location, double value){
            weights_to_next(row_location, column_location) = value;
        }

        void main_logic(Eigen::VectorXd input_data){
            /* main logic that calculates the output with the input and weights */

            if (input_data.size() != neuron_outputs.size()) cout << "we got problems" << endl;

                input = input_data;

                // apply sigmoid activation
                for (int k = 0; k<input_data.size(); k++){ 
                    input_data[k] = sigmoid(input_data[k]);
                    neuron_outputs[k] = input_data[k];
                }

                // calculate the output
                for (int i = 0; i<next_neuron_count; i++){
                    output[i] = input_data.dot(weights_to_next.col(i)); // assing the output as sum(the_data * the weight.col(i))
                }
                
        }
        Eigen::VectorXd get_output(){
            return output;
        }
        Eigen::VectorXd get_neuron_output(){
            return neuron_outputs;
        }

        Eigen::MatrixXd get_weights(){
            return weights_to_next;
        }
        Eigen::VectorXd get_input(){
            return input;
        }

        vector<double> get_ds(){
            return ds;
        }

        double get_ds(int index){
            return ds[index];
        }

        void insert_ds(int row, double value){
            ds[row] = value;
        }
};


class OutputLayer{

    private:
        const int neuron_count;
        Eigen::RowVectorXd input;
        double output;
        double threshold;
        vector<double> ds; // vector to hold neuron gradient (ds)

        double sigmoid_function(double input) {
            return (1.0 / (1.0 + exp(-input)));
        }
    
    public:
        
        // constructor
        OutputLayer(const int neuron_count, const int threshold) : neuron_count(neuron_count), threshold(threshold) {
            vector<double> ds_vec(neuron_count);
            ds = ds_vec;
        }

        // deconsturctor
        ~OutputLayer(){}

        void main_logic(Eigen::VectorXd input){
            /* Main logic that processes the input to output layer and writes the output to the output variable */

            input = input;
            double sum = input.sum();
            double decision_sum = sigmoid_function(sum);
            
            output = decision_sum >= threshold ? 1 : 0;
        }

        double get_output(){
            return output;
        }

        Eigen::RowVectorXd get_input(){
            return input;
        }

        vector<double> get_ds(){
            return ds;
        }

        void insert_ds(double value, double row = 0){
            ds[row] = value;
        }
};

class NeuralNetwork{

    private:
        InputLayer input_layer;
        vector<HiddenLayer> hiddens;
        OutputLayer output_layer;

        double calculate_error(double y_actual, double y_pred){ // wroted as a seperate function to ease the change if required
            return (y_actual - y_pred) * (y_actual - y_pred);
        }

    public:
        
        // constructor
        NeuralNetwork(InputLayer input_layer, vector<HiddenLayer> hiddens, OutputLayer output_layer) : input_layer(input_layer), hiddens(hiddens), 
        output_layer(output_layer) {
            cout << "nn initializing" << endl;
        }

        // deconstructor
        ~NeuralNetwork(){
            cout << "network deleted" << endl;
        }

        double run_one_sample(Eigen::VectorXd input){
            /* feeds one sample through the network */

            input_layer.main_logic(input);
            
            Eigen::VectorXd out_input = input_layer.get_output();
            Eigen::VectorXd out_hidden = out_input;
            
            for (int i = 0; i<hiddens.size(); i++){ // run for every hidden layer in hiddens vector

                hiddens[i].main_logic(out_hidden);
             
                out_hidden = hiddens[i].get_output();
            }
            
            output_layer.main_logic(out_hidden);
            
            double output = output_layer.get_output();
            
            return output;
        }

        
        void calculate_ds_output(OutputLayer& output_l, double y_actual){
            /* Calculates the gradient for  output layer : ds = (output - y_actual) * output * (1 - output)*/
            
            double output = output_l.get_output();
            double ds = (output - y_actual) * output * (1 - output);
            output_l.insert_ds(ds);
        }

        void calculate_ds_hidden(HiddenLayer& hidden_l, vector<double> next_l_ds){
            /* Calculates the gradient for given hidden layer : ds = output * (1 - output) * sum(weights * ds_next) */

            Eigen::MatrixXd hidden_weights = hidden_l.get_weights(); // cols() should be equal to next_l_ds.size() !IMPORTANT
            
            // get the output of each neuron to compute their ds by mutliplying their connected next layer neuron's ds
            Eigen::VectorXd hidden_outputs = hidden_l.get_neuron_output();
            
            for (int i = 0; i<hidden_weights.rows(); i++){

                double error_coming_from_next = 0;
                Eigen::RowVectorXd weights_from_one_neuron = hidden_weights.row(i);
                
                for (int k = 0; k<weights_from_one_neuron.size(); k++){
                    error_coming_from_next += weights_from_one_neuron[k] * next_l_ds[k];
                }
                double ds = (hidden_outputs[i] * (1 - hidden_outputs[i]) * error_coming_from_next);

                hidden_l.insert_ds(i, ds);
            }
        }

        void update_weights_hidden(HiddenLayer& hidden_l, vector<double> ds_next, double learning_rate){
            /* Updates the weigths in the hidden layer : new_weight -= learning_rate * output * ds_next */

            Eigen::VectorXd output = hidden_l.get_neuron_output();
        
            Eigen::MatrixXd hidden_weights = hidden_l.get_weights();
            
            for (int i = 0; i < hidden_weights.rows(); i++){

                Eigen::VectorXd weights_from_i = hidden_weights.row(i);

                for (int k = 0; k < weights_from_i.size(); k++){
                
                    double value = output[i] * ds_next[k];
                
                    double new_weight = weights_from_i[k] - learning_rate * value;
                    hidden_l.update_one_weight(k, i, new_weight);   
                }   
            }
        }


        void train(Eigen::MatrixXd data, Eigen::VectorXd y, const double learning_rate, const int max_iterations){
            /* train loop that applies stochastic gradient descent */

            for (int i = 0; i<data.rows(); i++){

                Eigen::VectorXd input_row = data.row(i);
                double y_actual = y[i];

                double y_pred = run_one_sample(input_row);
                
                double error = calculate_error(y_actual, y_pred);
                
                // calculate ds for output layer
                calculate_ds_output(output_layer, y_actual);

                vector<double> to_give = output_layer.get_ds();

                for (int k = hiddens.size() - 1; k>=0; k--){ // go backward
                    
                    // calculate ds for hidden layer
                    calculate_ds_hidden(hiddens[k], to_give);

                    // calculate and update new weights for hidden layer
                    update_weights_hidden(hiddens[k], to_give, learning_rate);
                    
                    to_give = hiddens[k].get_ds();
                
                }
                            
            }
            cout << "training done" << endl;
        }
};

Eigen::MatrixXd prepare_input_data(){
    Eigen::MatrixXd data(6,3);
    Eigen::RowVectorXd row1(3);
    row1 << 1,1,1;
    Eigen::RowVectorXd row2(3);
    row2 << 1,-1,1;
    Eigen::RowVectorXd row3(3);
    row3 << 0,-1,1;

    // adjusted training set
    Eigen::RowVectorXd row4(3);
    row4 << 1,1,-1;
    Eigen::RowVectorXd row5(3);
    row5 << 1,-1,-1;
    Eigen::RowVectorXd row6(3);
    row6 << 0,-1,-1;
    
    data.row(0) = row1;
    data.row(1) = row2;
    data.row(2) = row3;
    data.row(3) = row4;
    data.row(4) = row5;
    data.row(5) = row6;

    return data;
}

// to compile and run: g++ MultiLayerNN.cpp -o MultiLayerNN -I ../eigen-3.4.0/ && ./MultiLayerNN

int main(){

    // Create data
    Eigen::MatrixXd data = prepare_input_data();

    // Create output (y column, target variable)
    Eigen::VectorXd output(6);
    output << 1,1,0,0,1,0;

    // Initialize input layer with 3 neurons
    InputLayer input_layer(3, 4);
    input_layer.adjust_weights();

    // Initialize hidden layers below with 4 neurons each
    HiddenLayer first_layer(4,4);
    first_layer.adjust_weigths();

    HiddenLayer second_layer(4,4);
    second_layer.adjust_weigths();

    HiddenLayer third_layer(4,1);
    second_layer.adjust_weigths();

    // Initialize ouput layer wiht 1 neuron
    OutputLayer output_layer(1, 0);
 
    // get hidden layers in a vector
    vector<HiddenLayer> hiddens;
    hiddens.push_back(first_layer);
    hiddens.push_back(second_layer);

    // Initialize network to orchestrate the layers
    NeuralNetwork nn = NeuralNetwork(input_layer, hiddens, output_layer);
    
    // run the training
    double learning_rate = 0.001;
    int max_iterations = 1000;
    nn.train(data, output, learning_rate, max_iterations);

    return 0;
}