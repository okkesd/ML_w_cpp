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
        Eigen::RowVectorXd weights;
        int output;

    public:
        
        InputLayer(int neuron_count, int next_neuron_count) : neuron_count(neuron_count) {
            weights = Eigen::RowVectorXd(neuron_count);
        }

        void adjust_weights(Eigen::RowVectorXd weights){

            if (!weights.size() == weights.size()){
                return;
            }

            weights = weights;  
            cout << "weights adjusted" << endl;
        }

        void adjust_weights(){
            weights = Eigen::RowVectorXd::Ones(neuron_count);
        }
        
        void main_logic(Eigen::RowVectorXd input_data){  
            
            if (!input_data.rows() == neuron_count){
                return;
            }
            double sum = 0;
            for (int k = 0; k<input_data.size(); k++){
                sum += input_data[k] * weights[k];
            }
            output = sum;
        }

        int get_output(){
            return output;
        }

        Eigen::RowVectorXd get_weights(){
            return weights;
        }
        void print_weights(){
            
            cout << "Weights: " << weights << endl << endl;
        }

        int get_neuron_count(){
            return neuron_count;
        }
};

void train_perceptron(InputLayer& input_layer, const Eigen::MatrixXd& data, const int max_iterations){

    for (int j = 0; j<max_iterations; j++){
        int negative_output_count = 0;
        
        for (int i = 0; i<data.rows(); i++){

            Eigen::RowVectorXd one_row = data.row(i);
            input_layer.main_logic(one_row);

            cout << "i: " << i << " output: " << input_layer.get_output() << endl << endl;
            if (input_layer.get_output() <= 0){

                cout << "output is not positive, updating weights..." << endl;
                Eigen::RowVectorXd new_weights(input_layer.get_neuron_count());
                new_weights = (one_row * input_layer.get_weights());
                //for (int k = 0; k<one_row.size(); k++){
                //    new_weights[k] = one_row[k] + input_layer.get_weights()[k];
                //}
                
                input_layer.adjust_weights(new_weights);
                input_layer.print_weights();
                negative_output_count++;
            }
        
        }
        if (negative_output_count == 0){cout << "j is at the end (convergence)" << j << endl; break;}
    }
    cout << "\nTraining done, current weights: \n";
    input_layer.print_weights();
}

int main(){

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

    cout << "data\n" << data << endl;

    Eigen::VectorXd output(4);
    output << 1,1,0,0;

    InputLayer input_layer(3, 1);

    input_layer.adjust_weights();
    input_layer.print_weights();
    
    int max_iterations = 5;
    train_perceptron(input_layer, data, max_iterations);

    return 0;
}