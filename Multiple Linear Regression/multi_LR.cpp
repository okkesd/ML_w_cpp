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

        Eigen::RowVectorXd row(6);
        row(0) = 1.0; // bias term
        for (int j = 0; j < 5; ++j)
            row(j + 1) = values[j]; // features

        X_rows.push_back(row);
        y_values.push_back(values[5]); // last column is the target

        // put the row into data
        //data.push_back(Data_Row(stoi(row[0]), stoi(row[1]), stoi(row[2]), stoi(row[3]), stoi(row[4]), stoi(row[5]))); 
        //X.addTo(stoi(row[0]), stoi(row[1]), stoi(row[2]), stoi(row[3]), stoi(row[4]));
  
        i++;
    }
    // Convert vectors to Eigen matrix/vector
    int n_samples = X_rows.size();
    Eigen::MatrixXd X_temp(n_samples, 6); // 6 is for 1 bias and 5 features
    Eigen::VectorXd y_temp(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        X_temp.row(i) = X_rows[i];
        y_temp(i) = y_values[i];
    }

    X = X_temp;
    y = y_temp;
}

void fit_model(Eigen::VectorXd& theta, Eigen::MatrixXd& X, Eigen::VectorXd& y, double alpha, int max_iterations){

    int m = X.rows();
    int n = X.cols();

    double epsilon = 1e-9;

    double previous_cost = std::numeric_limits<double>::max();

    for (int i = 0; i<max_iterations; i++){

        // calculate the predictions for all of rows by multiplying X and theta
        Eigen::VectorXd predictions = X * theta; 

        // calculate errors for each rows
        Eigen::VectorXd errors = predictions - y; 

        // calculate the theta with gradient descent
        Eigen::VectorXd gradient  = (X.transpose() * errors) / m;

        // calculate the current cost
        double cost = (errors.array().square().sum()) / (2.0 * m); 

        // check if cost is close to the previous one or near 0
        if (abs(cost - previous_cost) < epsilon || (gradient.norm() < epsilon)) {

            std::cout << "Converged at iteration " << i << endl;
            cout << "Iteration " << i << " - Cost: " << cost << endl;
            break;
        }

        // update the theta and cost
        theta -= (alpha) * gradient;
        previous_cost = cost;

        if (i % 100 == 0) {    
            cout << "Iteration " << i << " - Cost: " << cost << endl;
        }
    }
}

pair<vector<double>, vector<double>> normalize_features(Eigen::MatrixXd& X) {

    // we will store the mean and std of every column
    vector<double> means(X.cols());
    vector<double> stddevs(X.cols());

    for (int i = 1; i < X.cols(); ++i) {  // skip column 0 if it's the bias term
        double mean = X.col(i).mean();
        double stddev = sqrt((X.col(i).array() - mean).square().sum() / X.rows());

        X.col(i) = (X.col(i).array() - mean) / stddev;

        means[i - 1] = mean;
        stddevs[i - 1] = stddev;
    }
    return make_pair(means, stddevs);
}

// to compile run, g++ multi_LR.cpp -o multi -I ../eigen-3.4.0

// todo: w ve b yi ayri tutup w'deki değerleri kendimiz atatadigimiz sistem kur (perhaps in new cpp file), train test split, 
int main(){

    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    
    read_csv("Student_Performance.csv", X, y);


    Eigen::VectorXd y_pred;
    Eigen::VectorXd theta = Eigen::VectorXd::Ones(6);  // 5 for features and 1 for bias
    double learning_rate = 0.00035;
    int max_iterations = 500000;

    auto [means, stddevs] = normalize_features(X);


    auto start = chrono::high_resolution_clock::now();

    fit_model(theta, X, y, learning_rate, max_iterations); // last is 2.13 with alpha=0.00035, iter=500000

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Time taken: " << duration << " miliseconds\n";

    cout << theta << endl;

    Eigen::RowVectorXd sample(6);
    sample << 1, 8, 70, 1, 7, 100; // 1 for bias, followed by the 5 features

    double y_pred_1 = sample * theta;
    cout <<"first y_pred is : " << y_pred_1 << endl;

    vector<int> raw_sample = {2, 54, 1, 4, 9};
    Eigen::RowVectorXd sample2(6);
    sample2(0) =  1; // 1 for bias, followed by the 5 features

    for (int i = 0; i<raw_sample.size(); i++){
        sample2(i+1) = (raw_sample[i] - means[i]) / stddevs[i];
    }
    
    double y_pred2 = sample2 * theta;
    cout <<"first y_pred is : " << y_pred2 << " while y is -> 30" << "   error is: " << abs(y_pred2 - 30) << endl;

    return 0;
}