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

class Deck{
public:
    Deck();
    Deck(double * values, int n, int m);
    Deck(int n, int m, double min, double max, int seed);
    
    void shuffle();
    Card** distribute(int players);
    
    Card* getCards();
    int getM();
    int getN();
    
    double getHV();
    double getSD();
    void computeRanks();
    
private:
    Card* cards;
    int n;
    int m;
};

#endif /* DECK_H */
