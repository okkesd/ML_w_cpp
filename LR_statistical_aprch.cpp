#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
//#include "matplotlibcpp.h"

//namespace plt = matplotlibcpp;
using namespace std;

template <typename T>
double mean(const std::vector<T>& vec) {
    if (vec.empty()) return 0.0;
    double sum = accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}



double calculate_S_x_x(vector<int> x){
    double sum = 0;
    double avg = mean(x);
    
    for (int i=0; i<x.size(); i++){
        sum += (x[i] - avg) * (x[i] - avg);
    }

    return sum;
}

double calculate_S_x_y(vector<pair<int, double>> veri){

    double x_avg = 0;
    double y_avg = 0;

    for (int i = 0; i<veri.size(); i++){
        x_avg += veri[i].first;
        y_avg += veri[i].second;
    }
    
    x_avg = x_avg / veri.size();
    y_avg = y_avg / veri.size();

    double sum = 0;
    double sum2 = 0;

    for (int i = 0; i<veri.size(); i++){
        sum += (veri[i].first - x_avg) * (veri[i].second - y_avg);
        sum2 += (veri[i].first - x_avg) * (veri[i].first - x_avg);
    }
    return static_cast<double>(sum/sum2);
}

vector<double> create_equation_and_predict(vector<pair<int, double>> veri, double Sxx, double Sxy){
    double x_avg = 0;
    double y_avg = 0;

    for (int i = 0; i<veri.size(); i++){
        x_avg += veri[i].first;
        y_avg += veri[i].second;
    }
    
    x_avg = x_avg / veri.size();
    y_avg = y_avg / veri.size();

    double coefficient = Sxy / Sxx;
    double constant = y_avg - coefficient * x_avg;
    cout << "equation : y ="<< coefficient <<"x + "<< constant << endl; 
    double error = 0;
    double error_kare = 0;
    double ort_kare = 0;

    vector<double> predictions;

    for (int i = 0; i<veri.size(); i++){
        double y_pred = constant + coefficient * veri[i].first;
        //cout << "x=" << veri[i].first << " icin y_pred: " << y_pred << " (actual: " << veri[i].getY() <<")" << endl;
        predictions.push_back(y_pred);
        error += fabs(y_pred - veri[i].second);
        error_kare += (y_pred - veri[i].second) * (y_pred - veri[i].second);
        ort_kare += (y_avg - veri[i].second) * (y_avg - veri[i].second);
    }

    cout << "Absolute error: " << error << endl;
    cout << "Mean absolute error: " << error/veri.size() << endl;
    cout << "Mean squarred error: " << error_kare/ 2 * veri.size() << endl;
    cout << "R2: " << 1 - (error_kare/ort_kare) << endl;
    return predictions;
}


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

/*
    TODO: 
        implement gradient descent somehow (maybe change the logic of regression to different ways)
        later, logistic regression

        intial is 1 *x + 1

        J(w,b) = 1/(2*m) * sum((w*x(i) +b - y(i)) ^ 2)

        derivative(J(w,b), w) = 1/m * sum((w*x(i) +b - y(i)) * x(i))

        derivative(J(w,b), b) = 1/m * sum(w*x(i) +b - y(i))
         
        temp_w = w - a * derivative(J(w,b), w) avec a est learning rate
        temp_b = b - a * derivative(J(w,b), b)

        w = temp_W
        b = temp_b
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

    vector<int> x_axis;
    for (int i = 1; i<store_data.size(); i++){
        x_axis.push_back(i);
    }

    double S_x_x = calculate_S_x_x(x_axis);

    cout << "Sxx: " << S_x_x << endl;

    double S_x_y = calculate_S_x_y(store_data);

    cout << "Sxy: " << S_x_y << endl;

    vector<double> predictions = create_equation_and_predict(store_data, S_x_x, S_x_y); // hold the predictions in here
    cout << "data_size: " << store_data.size() << endl;

    ofstream file("pred" + to_string(store_num) + ".csv");
    file << "store,x,y,y_pred\n";
    for (int i=0; i<store_data.size(); i++){

        file << store_num << "," << store_data[i].first << "," << store_data[i].second << "," << predictions[i] << endl;
    }
    file.close();

    string command = "python3 plot.py pred" + to_string(store_num) + ".csv";
    system(command.c_str()); // run the above command on command line to show the plot

    /*cout << endl << "*** Veri***" << endl;
    for (int i = 0; i < veri.size(); i++) {  
        cout << "( " << store_data[i].first << " - " <<  store_data[i].second << " )" << endl;
    }*/
    
    
    return 0;
}