#include <stdio.h>
#include <Eigen/Dense>

// sigmoid activation function 
double sigmoid_func(double input){
    return 1.0 / (1.0 + exp(-input));
}

// McCulloch - Pitts neuron with TLU
class Neuron{

    private:
        double threshold;
        int output;
        int state;
        
    public:

        // function to return the activation state of a neuron
        bool tlu(Eigen::VectorXd& x, Eigen::VectorXd& w){
            double sum = 0;

            for (int i = 0; i<x.size(); i++){
                sum += x[i] * w[i]; // add the x*w to the sum where x[i] can be 1 or 0 (or -1 if preferred)
            }
            this->state = sum > this->threshold ?  1 : 0;
            return sum > this->threshold;
        }

        // function to return the activation power of a neuron
        double tlu_continous(Eigen::VectorXd& x, Eigen::VectorXd& w){
            double sum = 0;
            for (int i = 0; i<x.size(); i++){
                sum += x[i] * w[i]; // add the x*w to the sum where x[i] can be 1 or 0 (or -1 if preferred)
            }
            return sigmoid_func(this->threshold - sum);
        }
};
// Update rule --> wij_new = wij_old + alpha * xi * yi where xi is the input to j from i and yi is the output of j

int main(){

    return 0;
}