#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <chrono>

using namespace std;

struct Data_Row{
    int hours;
    int previous_score;
    int extra_activity;
    int sleep_hours;
    int practice_questions;
    int performance;

    Data_Row(int hours, int previous_score, int extra_activity, int sleep_hours, int practice_questions, int performance) 
    :hours(hours),previous_score(previous_score),extra_activity(extra_activity), sleep_hours(sleep_hours), practice_questions(practice_questions),
     performance(performance){}
};

void read_csv(string path, Eigen::MatrixXd& X, Eigen::VectorXd& y){

    
    ifstream fin;
    
    fin.open(path, ios::in);
    int hours, previous_score, extra_activity, sleep_hours, practice_questions, performance;
    string line; // line holder

    vector<Data_Row> data;
    int i = 0;
    vector<Eigen::RowVectorXd> X_rows;
    vector<double> y_values;


    while (getline(fin, line)){ // get the line in line variable

        if (i==0) {i++; continue;} // skip the first line since they are headers
        
        stringstream ss(line);
        //vector<string> row;
        string cell;
        vector<int> values;

        while(getline(ss, cell, ',')){ // get every value between ',' into cell variable

            if (cell == "Yes"){
                cell = "1";
            } else if (cell == "No"){
                cell = "0";
            }

            //row.push_back(cell);
            values.push_back(stod(cell));
        }

        Eigen::RowVectorXd row(5);
        for (int j = 0; j < 5; ++j)
            row(j) = values[j]; // features

        X_rows.push_back(row);
        y_values.push_back(values[5]); // last column is the target

        // put the row into data
        //data.push_back(Data_Row(stoi(row[0]), stoi(row[1]), stoi(row[2]), stoi(row[3]), stoi(row[4]), stoi(row[5]))); 
        //X.addTo(stoi(row[0]), stoi(row[1]), stoi(row[2]), stoi(row[3]), stoi(row[4]));
  
        i++;
    }
    // Convert vectors to Eigen matrix/vector
    cout << X_rows.size() << endl;
    int n_samples = X_rows.size();
    Eigen::MatrixXd X_temp(n_samples, 5); // 6 is for 1 bias and 5 features
    Eigen::VectorXd y_temp(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        X_temp.row(i) = X_rows[i];
        y_temp(i) = y_values[i];
    }

    X = X_temp;
    y = y_temp;
}

void feature_engineering(Eigen::MatrixXd& X){

    Eigen::VectorXd studied_scor(X.rows());
    Eigen::VectorXd studied_sample(X.rows());
    Eigen::VectorXd scor_sample(X.rows());
    Eigen::VectorXd studied_scor_sample(X.rows());

    Eigen::MatrixXd new_X(X.rows(), X.cols()+4);
    
    for (int i = 0; i<X.rows(); i++){
        studied_scor(i) = X.col(0)(i) * X.col(1)(i);
        studied_sample(i) = X.col(0)(i) * X.col(4)(i);
        scor_sample(i) = X.col(1)(i) * X.col(4)(i);
        studied_scor_sample(i) = X.col(0)(i) * X.col(1)(i) * X.col(4)(i);
    }

    new_X.block(0,0, X.rows(), X.cols()) = X;

    new_X.col(X.cols()) = studied_scor;    
    new_X.col(X.cols()+1) = studied_sample;
    new_X.col(X.cols()+2) = scor_sample;
    new_X.col(X.cols()+3) = studied_scor_sample;
    
    X = new_X;
}

void train_test_split(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train, 
                      Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test, double test_size){

    if (!(0 < test_size && test_size < 1)){
        return;
    }
    double train_size = static_cast<double>(1 - test_size);

    X_train = X.topRows(train_size * X.rows());
    y_train = y.head(train_size * y.size());

    X_test = X.bottomRows(test_size * X.rows());
    y_test = y.tail(test_size * y.size());
}

void fit_model(double& constant,Eigen::VectorXd& theta, Eigen::MatrixXd& X, Eigen::VectorXd& y, double alpha, int max_iterations){

    int m = X.rows();
    int n = X.cols();

    double epsilon = 1e-9;

    double previous_cost = std::numeric_limits<double>::max();

    for (int i = 0; i<max_iterations; i++){

        Eigen::VectorXd b = Eigen::VectorXd::Ones(X.rows());

        // calculate the predictions for all of rows by multiplying X and theta
        Eigen::VectorXd predictions = X * theta + (b* constant);

        // calculate errors for each rows
        Eigen::VectorXd errors = predictions - y; 
        

        for (int i = 0; i<X.cols(); i++){
            double gradient_i = ((errors.array() * X.col(i).array()).sum())*1/X.rows();
            theta[i] -= alpha * gradient_i;
        }
        constant -= alpha * (errors.sum())* 1/X.rows();
        
        

        // calculate the theta with gradient descent
        //Eigen::VectorXd gradient  = (X.transpose() * errors) / m;

        // calculate the current cost
        double cost = (errors.array().square().sum()) / (2.0 * m); 

        // check if cost is close to the previous one or near 0
        if (abs(cost - previous_cost) < epsilon ) { // || (gradient.norm() < epsilon)

            std::cout << "Converged at iteration " << i << endl;
            cout << "Iteration " << i << " - Cost: " << cost << endl;
            break;
        }

        // update the theta and cost
        //theta -= (alpha) * gradient;
        previous_cost = cost;

        if (i % 1000 == 0) {    
            cout << "Iteration " << i << " - Cost: " << cost << endl;
        }
    }
}

void test_model(Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test, Eigen::VectorXd& theta, double constant){

    // calculate predictions on test data
    Eigen::VectorXd y_pred = X_test * theta + (Eigen::VectorXd::Ones(X_test.rows()) * constant);

    // calculate errors between predictions and actual data
    Eigen::VectorXd errors = y_pred - y_test;

    // calculate mse
    double mse = static_cast<double>(errors.array().square().sum() / errors.size()); // just making sure its double

    // calculate r2_score
    double ss_tot = (y_test.array() - y_test.mean()).square().sum();
    double ss_res = errors.array().square().sum();
    double r2_score = static_cast<double> (1 - ss_res/ss_tot);

    cout << "MSE: " << mse << endl;
    cout << "R2_score: " << r2_score << endl;
}

pair<vector<double>, vector<double>> normalize_features(Eigen::MatrixXd& X_train, Eigen::MatrixXd& X_test) {

    // we will store the mean and std of every column
    vector<double> means(X_train.cols());
    vector<double> stddevs(X_train.cols());

    for (int i = 0; i < X_train.cols(); i++) {  // skip column 0 if it's the bias term
        double mean = X_train.col(i).mean();
        double stddev = sqrt((X_train.col(i).array() - mean).square().sum() / X_train.rows());

        X_train.col(i) = (X_train.col(i).array() - mean) / stddev;
        X_test.col(i) = (X_test.col(i).array() - mean) / stddev;

        means[i] = mean;
        stddevs[i] = stddev;
        //cout << "i is: " << i << " mean: " << mean << " sttdev: " << stddev << endl;
    }
    for (int i = 0; i<X_test.cols(); i++){
        
    }
    return make_pair(means, stddevs);
}

// to compile run, g++ multi_seperate.cpp -o multi_seperate -I ../eigen-3.4.0/ && ./multi_seperate

// todo: train test split, get r2 for test data etc. 
int main(){

    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    
    read_csv("Student_Performance.csv", X, y);
    cout << "csv read"<< endl;

    feature_engineering(X);

    Eigen::MatrixXd X_train;
    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_train;
    Eigen::VectorXd y_test;

    double test_size = 0.25;
    
    train_test_split(X, y, X_train, y_train, X_test, y_test, test_size);

    Eigen::VectorXd y_pred;
    Eigen::VectorXd theta = Eigen::VectorXd::Ones(X.cols());  // 5 for features and 1 for bias
    double b = 1;
    double& constant = b;
    double learning_rate = 0.00037;
    int max_iterations = 150000;

    // normalize X_train and X_test
    auto [means, stddevs] = normalize_features(X_train, X_test); 

    // start fitting the model
    auto start = chrono::high_resolution_clock::now();

    fit_model(constant, theta, X_train, y_train, learning_rate, max_iterations); // last is 2.13 with alpha=0.00035, iter=500000

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Time taken: " << duration << " ms\n";

    cout << "Theta: " << theta << "\nConstant: " << constant << endl;
    
    // test model and get metrics printed
    test_model(X_test, y_test, theta, constant);

    /*Eigen::RowVectorXd sample(5);
    sample << 12, 88, 1, 7, 20; // 1 for bias, followed by the 5 features
    for (int i = 0; i<sample.size(); i++){
        sample(i) = (sample[i] - means[i]) / stddevs[i];
    }


    double y_pred_1 = sample * theta + constant;
    cout <<"first y_pred is : " << y_pred_1 << endl;
    
    vector<int> raw_sample = {2, 54, 1, 4, 9};
    Eigen::RowVectorXd sample2(5);
    
    for (int i = 0; i<raw_sample.size(); i++){
        sample2(i) = (raw_sample[i] - means[i]) / stddevs[i];
    }
    
    double y_pred2 = (sample2 * theta) + constant;
    cout <<"first y_pred is : " << y_pred2 << " while y is -> 30" << "   error is: " << abs(y_pred2 - 30) << endl;
    */

    return 0;
}