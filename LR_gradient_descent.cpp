#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>

using namespace std;


struct Custom_Type{
    vector< pair <int, pair<int, double> > > vec; // we want to hold pair( int: store_num, pair( int: x_value, double: weekly_sales))
};

Custom_Type read_csv(string path){

    ifstream fin;
    
    fin.open(path, ios::in);
    int Store, Date, Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment;
    string line; // line holder

    vector< pair< int, pair<int,double> >> data;
    int i = 0;

    int store_num = 1;
    int x_value = 1;

    while (getline(fin, line)){ // get the line in line variable

        if (i==0) {i++; continue;} // skip the first line since they are headers
        
        stringstream ss(line);
        vector<string> row;
        string cell;

        while(getline(ss, cell, ',')){ // get every value between ',' into cell variable

            row.push_back(cell);
        }

        if (store_num == stoi(row[0])){

            data.push_back({stoi(row[0]), {x_value, stod(row[2])}}); // get store and weekly_sales while putting the x_value as sort of time serie
            x_value++;
        } else {
            
            store_num = stoi(row[0]);
            x_value = 1;
            data.push_back({stoi(row[0]), {x_value, stod(row[2])}}); // get store and weekly_sales
            x_value++;
        }
        
        
        //cout << "store: " << data[i-1].first << " weekly_sales: " << data[i-1].second.second << endl;
        i++;
    }

    Custom_Type new_data;
    new_data.vec = data;
    return new_data;
}

pair<double, double> calculate_partial_derivs(vector<pair<int, pair<double, double>>> pred_data, double w, double b){

    double w_value = 0.0;
    double b_value = 0.0;
    for (int i = 0; i<pred_data.size(); i++){
        w_value += (w*pred_data[i].first + b - pred_data[i].second.first) * pred_data[i].first;               //(w*x(i) +b - y(i)) * x(i)
        b_value += (w*pred_data[i].first + b - pred_data[i].second.first);
    }

    return make_pair(w_value, b_value);
}

pair<double, double> calculate_temp(vector<pair<int, pair<double, double>>> pred_data, double coefficient, double constant, double learning_rate){

    auto [partial_w, partial_b] = calculate_partial_derivs(pred_data, coefficient, constant);
    double temp_w = coefficient - learning_rate * (1/pred_data.size()) * partial_w;
    double temp_b = constant - learning_rate * (1/pred_data.size()) * partial_b;

    return make_pair(temp_w, temp_b);
}

double calculate_cost(vector<pair<int, pair<double, double>>> pred_data){

    double error_sum = 0;
    for (int i = 0; i<pred_data.size(); i++){
        error_sum += (pred_data[i].second.second - pred_data[i].second.second) * (pred_data[i].second.second - pred_data[i].second.second);
    }

    return (1/(2*pred_data.size()) * error_sum);
}

vector<pair<int, pair<double,double>>> fit_model(vector<pair<int, double>> data){

    double coefficient = 1;
    double constant = 1;

    vector<pair<int, pair<double, double>>> pred_data; // (x, (y, y_pred))

    for (int i = 0; i<data.size(); i++){
        double y_pred = coefficient * data[i].first + constant;
        pred_data.push_back({data[i].first, {data[i].second, y_pred}});

    }

    double prev_cost = calculate_cost(pred_data);

    double learning_rate = 0.1;
    int max_iterations = 10000;

    for (int a = 0; a < max_iterations; a++) {
        
        auto [new_coefficient, new_constant] = calculate_temp(pred_data, coefficient, constant, learning_rate);
        
        coefficient = new_coefficient;
        constant = new_constant;        

        for (int i = 0; i<pred_data.size(); i++){ // write the new predictions
            pred_data[i].second.second = new_coefficient * pred_data[i].first + new_constant;
        }

        double current_cost = calculate_cost(pred_data);
        cout << "Cost: " << current_cost << endl;

        if (abs(prev_cost - current_cost) < 1e-6){
            break;
        } else {
            prev_cost = current_cost;
        }   
    }

    cout << "After fitting, best coefficient: " << coefficient << " , constant: " << constant << endl;

    return pred_data;
}

void print_error_metrics(vector<pair<int, pair<double, double>>> data){

    double error_kare = 0;
    double error = 0;
    double ort_kare = 0;
    double y_avg = 0;

    for (int i = 0; i<data.size(); i++){
        y_avg += data[i].second.first;
    }
    y_avg = y_avg / data.size();

    for (int i = 0; i<data.size(); i++){
        error_kare += (data[i].second.first - data[i].second.second) * (data[i].second.first - data[i].second.second);
        error += fabs(data[i].second.first - data[i].second.second);
        ort_kare += (y_avg - data[i].second.first) * (y_avg - data[i].second.first);
    }

    cout << "Mean Absolute Error: " << error / data.size() << endl;
    cout << "Mean Squared Error: " << error_kare / data.size() << endl;
    cout << "R2 score: " << 1 - error_kare/ort_kare << endl;
}

/*
    TODO:
        run the regression, evaluate and compare the results with statistical approach
*/

int main(){

    Custom_Type our_data;
    our_data = read_csv("Walmart_Sales.csv");
    vector< pair< int, pair<int, double> >> veri = our_data.vec; // it holds store and weekly_sales

    int store_num = 12;
    vector<pair<int, double>> store_data; // x and y

    for (auto row: veri){
        if (row.first == store_num){
            store_data.push_back({row.second.first, row.second.second});
        }
        
    }

    vector<pair<int, pair<double, double>>> fitted_data = fit_model(store_data);



    return 0;
}