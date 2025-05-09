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

void read_csv(string path, Eigen::MatrixXd& X, vector<Categorical>& X_categorical, vector<int>& y){ // Eigen::MatrixXd& X, Eigen::VectorXd& y

    
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
    y = y_values;
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
            {"MORTGAGE", 4},
            {"OTHER", 5}
        };
        row_to_add.segment(2, 4).setZero();
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
        row_to_add.segment(6,6).setZero();
        if (intent_map.count(row.loan_intent)){
            row_to_add[intent_map[row.loan_intent]] = 1;
        }

        // No Yes
        row.gender=="No" ? row_to_add[12] = 0 : row_to_add[12] = 1;

        new_categorical.row(i) = row_to_add;
    }
    return new_categorical;
}

double sigmoid_func(double input){
    double exp = 2.71;
    return 1/(1+pow(exp,(input * -1)));
}

// todo: person_income, loan_amount, credit score must be normalized, fit the model with gradient descent

// to compile and run: g++ Logistic_Regresssion.cpp -o logistic -I ../eigen-3.4.0/ && ./logistic

int main(){

    Eigen::MatrixXd X;
    vector<Categorical> X_categorical;
    vector<int> y;

    read_csv("./loan_data.csv", X, X_categorical, y);

    
    Eigen::MatrixXd new_X_categorical = encode_categorical(X_categorical);
    X_categorical.clear();

    Eigen::MatrixXd X_full(X.rows(), X.cols() + new_X_categorical.cols());
    X_full << X, new_X_categorical; // concatenates horizontally

    X.resize(2,2);
    new_X_categorical.resize(2,2);


    Eigen::VectorXd theta = Eigen::VectorXd::Ones(13);
    double b = 1;
    double& constant = b;
    
    return 0;
}