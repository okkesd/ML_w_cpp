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

class veri_cifti {
    public:
        int x;
        int y;

        veri_cifti(int x, int y) : x(x), y(y){};
    
        void print(){
            cout << x << ", " << y;
        }

        int getX(){
            return x;
        }

        int getY(){
            return y;
        }
};


double calculate_S_x_x(vector<int> x){
    double sum = 0;
    double avg = mean(x);
    
    for (int i=0; i<x.size(); i++){
        sum += (x[i] - avg) * (x[i] - avg);
    }

    return sum;
}

double calculate_S_x_y(vector<veri_cifti> veri){

    double x_avg = 0;
    double y_avg = 0;

    for (int i = 0; i<veri.size(); i++){
        x_avg += veri[i].getX();
        y_avg += veri[i].getY();
    }
    
    x_avg = x_avg / veri.size();
    y_avg = y_avg / veri.size();

    double sum = 0;
    for (int i = 0; i<veri.size(); i++){
        sum += (veri[i].getX() - x_avg) * (veri[i].getY() - y_avg);
    }
    return sum;
}

void create_equation_and_predict(vector<veri_cifti> veri, double Sxx, double Sxy){
    double x_avg = 0;
    double y_avg = 0;

    for (int i = 0; i<veri.size(); i++){
        x_avg += veri[i].getX();
        y_avg += veri[i].getY();
    }
    
    x_avg = x_avg / veri.size();
    y_avg = y_avg / veri.size();

    double coefficient = Sxy / Sxx;
    double constant = y_avg - coefficient * x_avg;
    cout << "equation : y ="<< coefficient <<"x + "<< constant << endl; 
    double error = 0;
    double error_kare = 0;
    double ort_kare = 0;

    for (int i = 0; i<veri.size(); i++){
        double y_pred = constant + coefficient * veri[i].getX();
        //cout << "x=" << veri[i].getX() << " icin y_pred: " << y_pred << " (actual: " << veri[i].getY() <<")" << endl;
        error += fabs(y_pred - veri[i].getY());
        error_kare += (y_pred - veri[i].getY()) * (y_pred - veri[i].getY());
        ort_kare += (y_avg - veri[i].getY()) * (y_avg - veri[i].getY());
    }

    cout << "Absolute error: " << error << endl;
    cout << "Mean absolute error: " << error/veri.size() << endl;
    cout << "Mean squarred error: " << error_kare/veri.size() << endl;
    cout << "R2: " << 1 - (error_kare/ort_kare) << endl;
}


struct Custom_Type{
    vector<pair<int, double>> vec;
};
Custom_Type read_csv(string path){

    ifstream fin;
    
    fin.open(path, ios::in);
    int Store, Date, Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment;
    string line; // line holder

    vector<pair<int,double>> data;
    int i = 0;

    while (getline(fin, line)){ // get the line in line variable

        if (i==0) {i++; continue;} // skip the first line since they are headers
        
        stringstream ss(line);
        vector<string> row;
        string cell;

        while(getline(ss, cell, ',')){ // get every value between ',' into cell variable

            row.push_back(cell);
        }

        data.push_back({stoi(row[0]), stod(row[2])});
        
        cout << "store: " << data[i-1].first << " weekly_sales: " << data[i-1].second << endl;
        i++;
    }

    Custom_Type new_data;
    new_data.vec = data;
    return new_data;
}

/*
    TODO: 
        spare the data in different stores, add time value, create the regression for each store
*/
int main(){

    Custom_Type our_data;
    our_data = read_csv("Walmart_Sales.csv");
    vector<pair<int, double>> new_veri = our_data.vec;

    vector<veri_cifti> veri;

    veri.push_back(veri_cifti(2,5));
    veri.push_back(veri_cifti(6,7));
    veri.push_back(veri_cifti(8,9));
    veri.push_back(veri_cifti(10,16));
    veri.push_back(veri_cifti(13,19));

    vector<int> x_axis;

    for (int i = 0; i<veri.size(); i++){
        x_axis.push_back(veri[i].getX());

    }   

    double S_x_x = calculate_S_x_x(x_axis);

    cout << "Sxx: " << S_x_x << endl;

    double S_x_y = calculate_S_x_y(veri);

    cout << "Sxy: " << S_x_y << endl;

    create_equation_and_predict(veri, S_x_x, S_x_y);

    cout << endl << "*** Veri***" << endl;
    for (int i = 0; i < veri.size(); i++) {
        
        veri[i].print();
        cout << endl;    
    }
    
    
    return 0;
}