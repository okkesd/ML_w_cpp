#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <chrono>
#include <string>
#include <map>

using namespace std;

struct Data_Row{
    
    float interest_rate, loan_percent_income;
    int age, income, emp_exp, loan_amnt, credit_hist_length, credit_score, loan_status;
    string gender, education, home_own, loan_intent, prev_loan_file;

    Data_Row(int age, string gender, string education, int income, int emp_exp, string home_own, int loan_amount, string loan_intent, float interest_rate, 
        float loan_percent_income, int credit_hist_length, int credit_score, string prev_loan_file, int loan_status): age(age), gender(gender), education(education),
        income(income), emp_exp(emp_exp), home_own(home_own), loan_intent(loan_intent), interest_rate(interest_rate), loan_percent_income(loan_percent_income),
        credit_hist_length(credit_hist_length), credit_score(credit_score), prev_loan_file(prev_loan_file), loan_status(loan_status) {}

};

struct Categorical{
    string gender, education, home_own, loan_intent, prev_loan_file;

    Categorical(string gender, string education, string home_own, string loan_intent, string prev_loan_file): 
    gender(gender), education(education), home_own(home_own), loan_intent(loan_intent), prev_loan_file(prev_loan_file) {}
};

void read_csv(string path, Eigen::MatrixXd& X, vector<Categorical>& X_categorical, Eigen::VectorXd& y){ // Eigen::MatrixXd& X, Eigen::VectorXd& y

    
    ifstream fin;
    
    fin.open(path, ios::in);
    float interest_rate, loan_percent_income;
    int age, income, emp_exp, loan_amnt, credit_hist_length, credit_score, loan_status;
    string gender, education, home_own, loan_intent, prev_loan_file;

    //vector<Data_Row> data;
    
    string line;
    vector<Eigen::RowVectorXd> X_rows;
    vector<Categorical> X_categorial_temp;
    vector<int> y_values;

    int i = 0;  
    cout << "on commence" << endl;

    while(getline(fin, line)){
        
        if (i==0) {i++; continue;}

        stringstream ss(line);
        string cell;
        vector<string> row;

        int j = 0;
        while(getline(ss, cell, ',')){

            ostringstream s;
            s << cell;
            row.push_back(s.str());
            j++;
        }
        
        age = stoi(row[0]);
        gender = row[1];
        education = row[2];
        income = stoi(row[3]);
        emp_exp = stoi(row[4]);
        home_own = row[5];
        loan_amnt = stoi(row[6]);
        loan_intent = row[7];
        interest_rate = stod(row[8]);
        loan_percent_income = stod(row[9]);
        credit_hist_length = stoi(row[10]);
        credit_score = stoi(row[11]);
        prev_loan_file = row[12];
        loan_status = stoi(row[13]);

        //cout << age << " " << gender << " " << interest_rate << " "<< loan_status << endl; 

        // Add categoricals
        Categorical categ(gender, education, home_own, loan_intent, prev_loan_file);
        X_categorial_temp.push_back(categ);
        
        Eigen::RowVectorXd row_vec(8);
        row_vec(0) = age;
        row_vec(1) = income;    
        row_vec(2) = emp_exp;
        row_vec(3) = loan_amnt;
        row_vec(4) = interest_rate;
        row_vec(5) = loan_percent_income;
        row_vec(6) = credit_hist_length;
        row_vec(7) = credit_score;

        y_values.push_back(loan_status);
        
        X_rows.push_back(row_vec);
        i++;
        //cout << i << endl;
    }
    

    Eigen::MatrixXd X_temp(X_rows.size(), X_rows[0].cols());
    
    for (int i = 1; i<X_rows.size(); i++){
        
        X_temp.row(i) = X_rows[i];
        //cout<<  X_rows[i] << endl;
    }
    
    X = X_temp;
    X_categorical = X_categorial_temp;
    
    y.conservativeResize(y_values.size());
    
    for (int i = 0; i<y_values.size(); i++){
        y(i) = y_values[i];
    }
    cout << "Done Reading csv" << endl;
}

Eigen::MatrixXd encode_categorical(const vector<Categorical>& X_categorical){

    // gender(2), prev_loan_file(2) -> binary encoding (1 + 1 = 2 columns)
    // home_own(4), loan_intent(6); -> One Hot encoding (4 + 6 = 10 columns)
    // education(5)                 -> 1 to n (ordinal encoding) (1 column)

    vector<string> unique_home_own;
    vector<string> unique_gender;
    vector<string> unique_loan_intent;
    vector<string> unique_prev_loan;
    vector<string> unique_education;

    for (int i = 0; i<X_categorical.size(); i++){
        int cnt_home = count(unique_home_own.begin(), unique_home_own.end(), X_categorical[i].home_own);
        if (!(cnt_home > 0)){
            unique_home_own.push_back(X_categorical[i].home_own);
        }

        int cnt_gender = count(unique_gender.begin(), unique_gender.end(), X_categorical[i].gender);
        if (!(cnt_gender > 0)){
            unique_gender.push_back(X_categorical[i].gender);
        }

        int cnt_loan_intent = count(unique_loan_intent.begin(), unique_loan_intent.end(), X_categorical[i].loan_intent);
        if (!(cnt_loan_intent > 0)){
            unique_loan_intent.push_back(X_categorical[i].loan_intent);
        }

        int cnt_prev = count(unique_prev_loan.begin(), unique_prev_loan.end(), X_categorical[i].prev_loan_file);
        if (!(cnt_prev > 0)){
            unique_prev_loan.push_back(X_categorical[i].prev_loan_file);
        }

        int cnt_edu = count(unique_education.begin(), unique_education.end(), X_categorical[i].education);
        if (!(cnt_edu > 0)){
            unique_education.push_back(X_categorical[i].education);
        }
    }
    cout << "Gender count: " << unique_gender.size() << endl;
    cout << "Home own count: " << unique_home_own.size() << endl;
    cout << "Loan intent count: " << unique_loan_intent.size() << endl;
    cout << "Prev loan file count: " << unique_prev_loan.size() << endl;
    cout << "Education count: " << unique_education.size() << endl;
    for (int i = 0; i<unique_prev_loan.size(); i++){
        //cout << unique_prev_loan[i] << " ";
    }
    
    Eigen::MatrixXd new_categorical(X_categorical.size(), 13); // 13 column (0-12)
    
    for (int i = 0; i<X_categorical.size(); i++){
        Categorical row = X_categorical[i];
        
        Eigen::RowVectorXd row_to_add(13);

        row.gender=="female" ? row_to_add[0] = 0 : row_to_add[0] = 1;

        // Master, High School, Bachelor, Associate, Doctorate
        map<string, int> education_map = {
            {"Doctorate", 4},
            {"Master", 3},
            {"Bachelor", 2},
            {"Associate", 1},
            {"High School", 0}
        };
        if (education_map.count(row.education)){
            row_to_add[1] = education_map[row.education];
        } else {
            cout << "erorrrrr on: "<< i << endl;
        }
        

        // RENT OWN MORTGAGE OTHER
        map<string, int> home_own_map = {
            {"RENT", 2},
            {"OWN", 3},
            {"MORTGAGE", 4} // OTHER for all 0
        };
        row_to_add.segment(2, 3).setZero();
        if (home_own_map.count(row.home_own)) {
            row_to_add[home_own_map[row.home_own]] = 1;
        }

        // PERSONAL EDUCATION MEDICAL VENTURE HOMEIMPROVEMENT DEBTCONSOLIDATION
        map<string, int> intent_map = {
            {"PERSONAL", 6},
            {"EDUCATION", 7},
            {"MEDICAL", 8},
            {"VENTURE", 9},
            {"HOMEIMPROVEMENT", 10},
            {"DEBTCONSOLIDATION", 11},
        };
        row_to_add.segment(5,6).setZero();
        if (intent_map.count(row.loan_intent)){
            row_to_add[intent_map[row.loan_intent]] = 1;
        }

        // No Yes
        row.gender=="No" ? row_to_add[12] = 0 : row_to_add[12] = 1;

        new_categorical.row(i) = row_to_add;
    }
    cout << "done encoding" << endl;
    return new_categorical;
}

vector<pair<double, double>> normalize_features(Eigen::MatrixXd& X, const vector<int> columns_to_normalize){

    vector<pair<double, double>> mean_stddev;
    
    for (int i = 0; i<columns_to_normalize.size(); i++){

        int col_num = columns_to_normalize[i];

        double mean = X.col(col_num).mean();
        double stddev = sqrt((X.col(col_num).array() - mean).square().sum() /X.rows());

        X.col(col_num) = (X.col(col_num).array() - mean) / stddev;

        mean_stddev.push_back(make_pair(mean, stddev));
        
    }
    cout << "Done normalize features" << endl;
    return mean_stddev;
}

void train_test_split(Eigen::MatrixXd& X,Eigen::VectorXd& y, Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train, 
                        Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test, double test_size=0.2){

    double train_size = 1.0 - test_size;
    int train_row_number = static_cast<int>(X.rows() * train_size);
    int test_row_number = X.rows() - train_row_number;

    /*cout <<"so far so good" << endl;
    X_train.resize(train_row_number, X.cols());
    y_train.resize(train_row_number);

    X_test.resize(test_row_number, X.cols());
    y_test.resize(test_row_number);*/

    cout << "resize done" << endl;
    X_train = X.topRows(train_row_number);
    y_train = y.head(train_row_number);

    X_test = X.bottomRows(test_row_number);
    y_test = y.tail(test_row_number);

    cout << "done train test split" << endl;
}

double sigmoid_func(double input);

void fit_model(Eigen::MatrixXd& X, const Eigen::VectorXd& y, Eigen::VectorXd& theta, double& constant, double learning_rate, const int maximum_iterations){

    double previous_cost = numeric_limits<double>::max();; // assign to maximum double number 
    double epsilon = 1e-6;

    for (int i = 0; i<maximum_iterations; i++){

        Eigen::VectorXd b = Eigen::VectorXd::Ones(X.rows());
        
        Eigen::VectorXd y_pred = X * theta + (b * constant);
        
        Eigen::VectorXd predictions(y_pred.rows());
        
        for (int i = 0; i<y_pred.rows(); i++){
        
            predictions(i) = sigmoid_func(y_pred(i));
        }
        
        // other version of loss: loss = -y * log(predictions) - (1 - y)log(1 - predictions) and cost is: cost = loss.array().sum() / loss.size()
        
        Eigen::VectorXd loss(predictions.size());
        
        loss = (predictions - y).array().square() / 2.0;
        
        for (int i = 0; i<loss.size(); i++){
        
            loss(i) = y(i) == 1 ? -log(predictions(i)) : -log(1.0 - predictions(i));
        }
        
        double current_cost = loss.array().sum() / loss.size();
        
        Eigen::VectorXd new_theta;
        for (int i = 0; i<X.cols(); i++){
        
            theta[i] -= learning_rate* ( ( (predictions - y).transpose() * X.col(i) ).sum()/ X.rows() );
        }
        constant -= learning_rate* ( (predictions - y).sum() / X.rows() );


        if (abs(current_cost - previous_cost) < epsilon){

            std::cout << "Converged at iteration " << i << endl;
            cout << "Iteration " << i << " - Cost: " << current_cost << endl;
            break;
        }

        // if not breaking, update the cost
        previous_cost = current_cost;

        if (i%100 == 0){
            cout << "Iteration " << i << " - Cost: " << current_cost << endl;
        }
    }
    
    cout << "Done fit model" << endl;
}

double sigmoid_func(double input){
    return 1.0 / (1.0 + exp(-input));
}

void test_model(Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test, const Eigen::VectorXd& theta, const double constant){

    Eigen::VectorXd b = Eigen::VectorXd::Ones(X_test.rows());
    Eigen::VectorXd predictions = X_test * theta + (constant * b);

    double mse = (predictions - y_test).array().square().sum() / predictions.size();
    cout << "MSE is: " << mse << endl;
    cout << "done test model" << endl;
}

// todo: train test split, cross fold evaluation

// to compile and run: g++ Logistic_Regresssion.cpp -o logistic -I ../eigen-3.4.0/ && ./logistic

int main(){

    Eigen::MatrixXd X; 
    vector<Categorical> X_categorical;
    Eigen::VectorXd y;

    // X columns: age, income, emp_exp, loan_amnt, interese_rate, loan_percent_income, credit_hist_length, credit_score
    read_csv("./loan_data.csv", X, X_categorical, y); 


    vector<int> columns_to_normalize = {1, 3, 7};
    vector<pair<double, double>> means_stddevs = normalize_features(X, columns_to_normalize);

    Eigen::MatrixXd new_X_categorical = encode_categorical(X_categorical);
    X_categorical.clear();


    Eigen::MatrixXd X_full(X.rows(), X.cols() + new_X_categorical.cols());
    X_full << X, new_X_categorical; // concatenates horizontally

    X.resize(2,2);
    new_X_categorical.resize(2,2);

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_full.cols());
    cout << theta.size() << endl;
    double b = 1;
    double& constant = b;
    double learning_rate = 0.001;
    int max_iterations = 1000;
    double test_size = 0.25;

    // train test split
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;

    Eigen::MatrixXd X_test;
    Eigen::VectorXd y_test;

    train_test_split(X_full, y, X_train, y_train, X_test, y_test, test_size);


    // fit model
    fit_model(X_train, y_train, theta, constant, learning_rate, max_iterations);

    test_model(X_test, y_test, theta, constant);

    cout << "theta is\n" << theta << endl;

    
    return 0;
}