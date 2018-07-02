/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Deck.h
 * Author: volz
 *
 * Created on 20. Juni 2018, 12:43
 */

#ifndef DECK_H
#define DECK_H

#include "Card.h"
#include <random>
#include <vector>

class Deck{
public:
    Deck();
    Deck(std::vector<double> values, int n, int m, std::vector<double> min, std::vector<double> max);
    Deck(int n, int m, double min, double max, int seed);
    
    void shuffle();
    std::vector<std::vector<Card>> distribute(int players);
    
    std::vector<Card> getCards();
    int getM();
    int getN();
    std::vector<double> getMin();
    std::vector<double> getMax();
    
    double getHV();
    double getSD();
    void computeRanks();
    
private:
    std::vector<Card> cards;
    int n;
    int m;
    std::vector<double> min;
    std::vector<double> max;
};

#endif /* DECK_H */
