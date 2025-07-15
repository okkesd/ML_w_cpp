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

        Layer(Eigen::MatrixXd weights_to_next) : weights_to_next(weights_to_next) {}
        Layer(){}

        Eigen::MatrixXd get_weights(){
            return weights_to_next;
        }

        virtual Eigen::VectorXd get_input(){
            return Eigen::VectorXd(); 
        }

        virtual void update_one_weight(int column_location, int row_location, double value){}

        virtual Eigen::VectorXd get_ds(){}
};

class InputLayer{
    private:
        int neuron_count;
        //int next_neuron_count;
        Eigen::VectorXd input;
        /*Eigen::MatrixXd weights_to_next; // the first row represents weights from firt neuron to the next layer
                                         // the firts column represents weights to the first neuron in the next layer*/

        Eigen::VectorXd output;

    public:
        
        // constructor
        InputLayer(int neuron_count) : neuron_count(neuron_count) {

            //weights_to_next = Eigen::MatrixXd(neuron_count, next_neuron_count); // weights should be from neurons in input to the neurons in the next layer
            output = Eigen::VectorXd(neuron_count); // neuron_count is rows, we will store the output of one neuron in a row of matrix
        }

        // destructor
        ~InputLayer(){}

        /*void adjust_weights(Eigen::RowVectorXd weights){

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
        */
        void main_logic(Eigen::VectorXd input_data){  
            /* main logic to process input via predefined weights */
            
            if (input_data.size() != neuron_count){
                return;
            }

            input = input_data;

            // calcualte the output
            output = input_data;
                
        }
        Eigen::VectorXd get_output(){
            return output;
        }

        /*Eigen::MatrixXd get_weights(){
            return weights_to_next;
        }
        void print_weights(){
            
            cout << "Weights: " << weights_to_next << endl;
        }*/

        int get_neuron_count(){
            return neuron_count;
        }
};

class HiddenLayer : public Layer{

    private:
        const int neuron_count;
        const int previous_neuron_count;
        
        Eigen::VectorXd input;
        Eigen::MatrixXd weights_to_next; // the first row represents the weights from the first neuron to the next layer
                                         // the first column represents the weights to the first neuron in the next layer

        Eigen::VectorXd output;          // the first element represents the input to the next layer's first neuron

        //Eigen::VectorXd neuron_outputs;  // the first element represents the output of the activation function of the first neuron

        Eigen::VectorXd ds;               // to hold the gradients (ds) of layer

        double sigmoid(double input){
            return (1.0 / (1.0 + exp(-input)));
        }
        Eigen::VectorXd sigmoid(Eigen::VectorXd input){

            Eigen::VectorXd another(input.size());
            for (int i = 0; i<input.size(); i++){
                another[i] = (1.0 / (1.0 + exp( -input[i] )));
            }
            return another;
        }
    
    public:

        HiddenLayer(const int previous_neuron_count, const int neuron_count) : neuron_count(neuron_count),
        previous_neuron_count(previous_neuron_count) {
        
            weights_to_next = Eigen::MatrixXd(previous_neuron_count, neuron_count);
            
            Layer(weights_to_next);

            output = Eigen::VectorXd(neuron_count);
            //neuron_outputs = Eigen::VectorXd::Zero(neuron_count);
            ds = Eigen::VectorXd::Zero(neuron_count);
        }

        void adjust_weigths(Eigen::VectorXd weights){
            weights_to_next = weights;
        }

        void adjust_weigths(){
            weights_to_next = Eigen::MatrixXd::Ones(previous_neuron_count, neuron_count);
        }

        void update_one_weight(int column_location, int row_location, double value) override {
            weights_to_next(row_location, column_location) = value;
        }

        void main_logic(Eigen::VectorXd input_data){
            /* main logic that calculates the output with the input and weights */

            if (input_data.size() != previous_neuron_count) cout << "we got problems" << endl;

                input = input_data;

                // calculate the output
                for (int i = 0; i<neuron_count; i++){
                    output[i] = sigmoid(input_data.dot(weights_to_next.col(i))); // assing the output as sum(the_data * the weight.col(i))
                }
        }
        Eigen::VectorXd get_output(){
            return output;
        }
        /*Eigen::VectorXd get_neuron_output(){
            return neuron_outputs;
        }*/

        Eigen::MatrixXd get_weights(){
            return weights_to_next;
        }
        Eigen::VectorXd get_input() override{
            return input;
        }

        Eigen::VectorXd get_ds() override{
            return ds;
        }

        double get_ds(int index){
            return ds[index];
        }

        void insert_ds(int row, double value){
            ds[row] = value;
        }
};


class OutputLayer : public Layer{

    private:
        const int neuron_count;
        const int previous_neuron_count;
        Eigen::VectorXd input;
        Eigen::MatrixXd weights_to_layer;
        Eigen::VectorXd output;
        double threshold;
        Eigen::VectorXd ds; // vector to hold neuron gradient (ds)

        double sigmoid_function(double input) {
            return (1.0 / (1.0 + exp(-input)));
        }
    
    public:
        
        // constructor
        OutputLayer(const int previous_neuron_count, const int neuron_count, const int threshold) : neuron_count(neuron_count), 
        previous_neuron_count(previous_neuron_count), threshold(threshold) {

            Eigen::VectorXd ds_vec(neuron_count);
            ds = ds_vec;
            weights_to_layer = Eigen::MatrixXd(previous_neuron_count, neuron_count);
            Layer(weights_to_next);
            output = Eigen::VectorXd::Ones(neuron_count);
        }

        // deconsturctor
        ~OutputLayer(){}

        void main_logic(Eigen::VectorXd input_row){
            /* Main logic that processes the input to output layer and writes the output to the output variable */

            input = input_row;
            
            for (int i = 0; i<neuron_count; i++){

                output[i] = sigmoid_function(input_row.dot(weights_to_layer.col(i)));
            }
            
            /*double sum = input.sum();
            double decision_sum = sigmoid_function(sum);
            
            output = decision_sum >= threshold ? 1 : 0;*/
        }

        Eigen::MatrixXd get_weights(){
            return weights_to_layer;
        }

        Eigen::VectorXd get_output(){
            return output;
        }

        Eigen::VectorXd get_input() override{
            return input;
        }

        Eigen::VectorXd get_ds() override{
            return ds;
        }

        void update_one_weight(int column, int row, double value) override{
            weights_to_layer(row, column) = value;
        }

        void insert_ds(Eigen::VectorXd value){
            ds = value;
        }
};

class NeuralNetwork{

    private:
        InputLayer input_layer;
        vector<HiddenLayer> hiddens;
        OutputLayer output_layer;

        Eigen::VectorXd calculate_error(Eigen::VectorXd y_actual, Eigen::VectorXd y_pred){ // wroted as a seperate function to ease the change if required
            return (y_actual - y_pred).array().square();
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

        Eigen::VectorXd run_one_sample(Eigen::VectorXd input){
            /* feeds one sample through the network */

            input_layer.main_logic(input);
            
            Eigen::VectorXd out_input = input_layer.get_output();
            Eigen::VectorXd out_hidden = out_input;
            
            for (int i = 0; i<hiddens.size(); i++){ // run for every hidden layer in hiddens vector

                hiddens[i].main_logic(out_hidden);
             
                out_hidden = hiddens[i].get_output();
            }
            
            output_layer.main_logic(out_hidden);
            
            Eigen::VectorXd output = output_layer.get_output();
            
            return output;
        }

        
        void calculate_ds_output(OutputLayer& output_l, Eigen::VectorXd y_actual){
            /* Calculates the gradient for  output layer : ds = (output - y_actual) * output * (1 - output)*/
            
            Eigen::VectorXd output = output_l.get_output();
            Eigen::VectorXd y(y_actual.size());
            y = Eigen::Map<Eigen::VectorXd>(y_actual.data(), y_actual.size());

            Eigen::VectorXd ds = (output - y).array() * output.array() * (Eigen::VectorXd::Ones(output.size()) - output).array();
            output_l.insert_ds(ds);
            
        }

        void calculate_ds_hidden(HiddenLayer& hidden_l, Eigen::VectorXd next_l_ds, Eigen::MatrixXd weights_to_next){
            /* Calculates the gradient for given hidden layer : ds = output * (1 - output) * sum(weights * ds_next) */

            Eigen::MatrixXd hidden_weights = hidden_l.get_weights(); // cols() should be equal to next_l_ds.size() !IMPORTANT
            
            
            // get the output of each neuron to compute their ds by mutliplying their connected next layer neuron's ds
            Eigen::VectorXd hidden_outputs = hidden_l.get_output();
            
            //cout << "ONEMLI : wegihts_to_next.rows() :=> " << weights_to_next.rows()<< endl;
            for (int i = 0; i<weights_to_next.rows(); i++){

                double error_coming_from_next = 0;
                Eigen::RowVectorXd weights_from_one_neuron = weights_to_next.row(i);
                

                for (int k = 0; k<weights_from_one_neuron.size(); k++){
                
                    error_coming_from_next += weights_from_one_neuron[k] * next_l_ds[k];
                }
                
                double ds = (hidden_outputs[i] * (1 - hidden_outputs[i]) * error_coming_from_next);
                //cout << "here: " << i << " hidden outputs size: " << hidden_outputs.size() << endl;
                hidden_l.insert_ds(i, ds);
                //cout << "insert_ds ok" << endl;
            }
        }

        // this should take output layer or hidden layer then calculate the weights associated them
        void update_weights_hidden(Layer& layer, double learning_rate){ //  Eigen::VectorXd ds_next,
            /* Updates the weigths in the hidden layer : new_weight -= learning_rate * output * ds_next */

            //Eigen::VectorXd output = hidden_l.get_output();
            if (HiddenLayer* hidden_layer = dynamic_cast<HiddenLayer*>(&layer)){

                //HiddenLayer* hidden_layer = dynamic_cast<HiddenLayer*>(&layer);
                cout << "its hidden" << endl;
                Eigen::VectorXd input = hidden_layer->get_input();
        
                Eigen::MatrixXd hidden_weights = hidden_layer->get_weights();
                
                for (int i = 0; i < hidden_weights.rows(); i++){
    
                    Eigen::VectorXd weights_from_i = hidden_weights.row(i);
    
                    for (int k = 0; k < weights_from_i.size(); k++){
                    
                        double value = input[i] * hidden_layer->get_ds()[k];
                    
                        double new_weight = weights_from_i[k] - learning_rate * value;
                        hidden_layer->update_one_weight(k, i, new_weight);   
                    }   
            }
            } else {

                OutputLayer* output_l = dynamic_cast<OutputLayer*>(&layer);
                cout << "its output" << endl;
                Eigen::VectorXd input = output_l->get_input();
                
                cout << "we got input -> " << input.size() << endl;

                Eigen::MatrixXd hidden_weights = output_l->get_weights();
                
                for (int i = 0; i < hidden_weights.rows(); i++){
    
                    Eigen::VectorXd weights_from_i = hidden_weights.row(i);
    
                    for (int k = 0; k < weights_from_i.size(); k++){
                    
                        double value = input[i] * output_l->get_ds()[k];
                    
                        double new_weight = weights_from_i[k] - learning_rate * value;
                        output_l->update_one_weight(k, i, new_weight);   
                    }   
                }
            }
        }


        void train(Eigen::MatrixXd data, Eigen::MatrixXd y, const double learning_rate, const int max_iterations){
            /* train loop that applies stochastic gradient descent */

            for (int i = 0; i<data.rows(); i++){

                Eigen::VectorXd input_row = data.row(i);
                Eigen::VectorXd y_actual = y.row(i);

                
                Eigen::VectorXd y_pred = run_one_sample(input_row);
          
                Eigen::VectorXd error = calculate_error(y_actual, y_pred);

                // calculate ds for output layer
                calculate_ds_output(output_layer, y_actual);

                     
                Eigen::VectorXd ds_to_give = output_layer.get_ds();
                //Layer layer_to_give = output_layer;
                Eigen::MatrixXd weights_to_give = output_layer.get_weights();

                for (int k = hiddens.size() - 1; k>=0; k--){ // go backward
                    
                    // calculate ds for hidden layer
                    calculate_ds_hidden(hiddens[k], ds_to_give, weights_to_give); // give next layer so we can compute layer's ds
                    //cout << "ds calculation done..." << endl;
                    ds_to_give = hiddens[k].get_ds();
                    weights_to_give = hiddens[k].get_weights();
                    //cout << "hobbba: " << ds_to_give.size() << endl;
                    cout << "first phase done" << endl;   
                }
                cout << "updating weights..." << endl;
                

                // TODO: we need to upcast the output layer to the Layer class so we can give both to the function
                // calculate and update new weights for hidden layer
                update_weights_hidden(output_layer, learning_rate); // no need next layer, all info in layer is enough

                for (int j = 0; j<hiddens.size(); j++){
                    update_weights_hidden(hiddens[j], learning_rate);
                }
                break;
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

// to compile and run: g++ neural_network_later.cpp -o neural_network_later -I ../eigen-3.4.0/ && ./neural_network_later

int main(){

    // Create data
    Eigen::MatrixXd data = prepare_input_data();

    // Create output (y column, target variable)
    Eigen::MatrixXd output(6,2);
    output << 1,1,
              1,0,
              0,0,
              1,0,
              0,1,
              1,0;


    // Initialize input layer with 3 neurons
    InputLayer input_layer(3);
    //input_layer.adjust_weights();

    // Initialize hidden layers below with 4 neurons each
    HiddenLayer first_layer(3, 4);
    first_layer.adjust_weigths();
    
    
    HiddenLayer second_layer(4,5);
    second_layer.adjust_weigths();

    HiddenLayer third_layer(5,6);
    second_layer.adjust_weigths();

    // Initialize ouput layer wiht 2 neuron
    OutputLayer output_layer(6, 2, 0);
 
    // get hidden layers in a vector
    vector<HiddenLayer> hiddens;
    hiddens.push_back(first_layer);
    hiddens.push_back(second_layer);
    hiddens.push_back(third_layer);

    // Initialize network to orchestrate the layers
    NeuralNetwork nn = NeuralNetwork(input_layer, hiddens, output_layer);
    
    // run the training
    double learning_rate = 0.001;
    int max_iterations = 1000;
    nn.train(data, output, learning_rate, max_iterations);

    return 0;
}