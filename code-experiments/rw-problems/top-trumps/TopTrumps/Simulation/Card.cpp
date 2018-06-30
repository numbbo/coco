#include "Card.h"
#include <stdexcept>
#include <iostream>

Card::Card(){
    this->m=0;
    this->ranks = std::vector<int>(m);
}

Card::Card(std::vector<double> values, int m, int offset){
    this->m = m;
    this->values = std::vector<double>(m);
    for(int i=0; i<m; i++){
        this->values[i] = values[i+offset];
    }
    this->ranks = std::vector<int>(this->m);
}

std::vector<double> Card::getValues(){
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

std::vector<int> Card::getRanks(){
    return this->ranks;
}

int Card::getRank(int i){
    return this->ranks[i];
}