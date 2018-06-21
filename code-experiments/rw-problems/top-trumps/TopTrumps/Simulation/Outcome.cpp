#include "Outcome.h"
#include <iostream>

Outcome::Outcome(int games){
    this->fair=0;
    this->games=games;
    this->leadChange=0;
    this->trickDiff=0;
}

double Outcome::getFairAgg(){
    return((double)this->fair/this->games);
}

double Outcome::getLeadChangeAgg(){
    return((double)this->leadChange/this->games);
}

double Outcome::getTrickDiffAgg(){
    return((double)this->trickDiff/this->games);
}

void Outcome::betterPlayerWon(){
    this->fair++;
}

void Outcome::leadChanged(){
    this->leadChange++;
}

void Outcome::addTrickDiff(int diff){
    this->trickDiff+=diff;
}

void Outcome::print(){
    std::cout << "fair: " << this->fair << " leadChange: " << this->leadChange << " trickDiff: " <<this->trickDiff << std::endl;
    std::cout << "fair: " << this->getFairAgg() << " leadChange: " <<this->getLeadChangeAgg() << " trickDiff: " <<this->getTrickDiffAgg()<< std::endl;
}