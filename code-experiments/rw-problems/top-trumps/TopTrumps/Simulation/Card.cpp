#include "Card.h"
#include <stdexcept>
#include <iostream>

Card::Card(){
    this->m=0;
    this->ranks = new int[this->m];
}

Card::Card(double* values, int m){
    this->m = m;
    this->values = new double[m];
    for(int i=0; i<m; i++){
        this->values[i] = *(values + (i));
    }
    this->ranks = new int[this->m];
}

double* Card::getValues(){
    return this->values;
}

double Card::getValue(int i){
    if(i<0 || i>=this->m){
        throw std::invalid_argument("no such value");
    }else{
        return this->values[i];
    }
}

void Card::toString(){
    std::cout << "Card:";
    for(int i=0; i<this->m; i++){
        std::cout << " " <<  this->values[i];
    }
    std::cout << std::endl;
}

void Card::printRanks(){
    std::cout << "Card ranks:";
    for(int i=0; i<this->m; i++){
        std::cout << " " <<  this->ranks[i];
    }
    std::cout << std::endl;
}

/*void Card::updateRanks(int* ranks){
    this-> ranks = ranks;
}*/

void Card::setRank(int rank, int j){
    this->ranks[j]=rank;
}

void Card::decreaseRank(int i){
    this->ranks[i]--;
}

int * Card::getRanks(){
    return this->ranks;
}

int Card::getRank(int i){
    return this->ranks[i];
}