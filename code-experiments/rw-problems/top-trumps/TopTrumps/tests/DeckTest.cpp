/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DeckTest.cpp
 * Author: volz
 *
 * Created on 20. Juni 2018, 14:06
 */

#include <stdlib.h>
#include <iostream>

#include "../Simulation/Deck.h"
#include "../Simulation/Card.h"
#include "../utils/hoy.h"
/*
 * Simple C++ Test Suite
 */

void createDeckDefault() {
    int n = 2;
    int m = 4;
    double * values = new double[n*m];
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    Deck deck(values, n, m);
    Card * cards = deck.getCards();
    for(int i=0; i<n; i++){
        cards[i].toString();
    }
    std::cout << "Expected outcome" << "Card: 0, 1, 2, 3 and Card: 4,5,6,7" << std::endl;
}

void createDeckRandom(){
    int n=2;
    int m=4;
    double min = 1;
    double max = 10;
    Deck deck(n, m, min, max, 1);
    Card * cards = deck.getCards();
    for(int i=0; i<n; i++){
        cards[i].toString();
    }
    std::cout << "Expected outcome: Two cards with 4 values ranging from 1-10" << std::endl;
    Deck deck3(n, m, min, max, 1);
    Card * cards3 = deck3.getCards();
    for(int i=0; i<n; i++){
        cards3[i].toString();
    }
    std::cout << "Expected outcome: Two cards with 4 values ranging from 1-10" << std::endl;
    Deck deck2(n, m, min, max, 2);
    Card * cards2 = deck2.getCards();
    for(int i=0; i<n; i++){
        cards2[i].toString();
    }
    std::cout << "Expected outcome: Different cards with 4 values ranging from 1-10" << std::endl;
}

void testShuffle(){
    int n = 10;
    int m = 4;
    double * values = new double[n*m];
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    Deck deck(values, n, m);
    deck.shuffle();
    Card * cards = deck.getCards();
    for(int i=0; i<n; i++){
        cards[i].toString();
    }
    std::cout << "Expected outcome: 10 cards with increasing values, but reshuffled" << std::endl;
    deck.shuffle();
    Card * cards2 = deck.getCards();
    for(int i=0; i<n; i++){
        cards2[i].toString();
    }
    std::cout << "Expected outcome: 10 cards with increasing values, but rereshuffled" << std::endl;
}

void testDistribute(){
    int n = 10;
    int m = 4;
    int players = 2;
    double * values = new double[n*m];
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    Deck deck(values, n, m);
    Card ** cards = deck.distribute(players);
    for(int i=0; i<players; i++){
        std::cout << "Player " << i << std::endl;
        Card * hand = cards[i];
        for(int j=0; j<n/players; j++){
            hand[j].toString();
        }
    }
    std::cout << "Expected outcome: 10 cards with increasing value, distributed to 2 players" << std::endl;
}


void testHV(){
    int n = 10;
    int m = 4;
    int players = 2;
    double * values = new double[n*m];
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    double refPoint[4]={37, 38, 39, 40};
    HVCalculator hv;
    std::cout << "Hypervolume: " << hv.computeHV(m,n,values,refPoint) << std::endl;
    Deck deck(values, n,m);
    std::cout << "Hypervolume: " << deck.getHV() << std::endl;
}

void testSD(){
    int n = 10;
    int m = 4;
    double * values = new double[n*m];
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    Deck deck(values, n,m);
    std::cout << "SD goal: " << deck.getSD() << std::endl;
    //Tested with R 1.290994
}

void testRanks(){
    int n = 10;
    int m = 4;
    Deck deck(n, m, 0, 10, 5);
    Card *cards = deck.getCards();
    for(int i=0; i<n; i++){
        cards[i].toString();
        cards[i].printRanks();
    }
}

int main(int argc, char** argv) {
    createDeckDefault();
    std::cout << "----------------------------------------" <<std::endl;
    createDeckRandom();
    std::cout << "----------------------------------------" <<std::endl;
    testShuffle();
    std::cout << "----------------------------------------" <<std::endl;
    testDistribute();
    std::cout << "----------------------------------------" <<std::endl;
    testHV();
    std::cout << "----------------------------------------" <<std::endl;
    testSD();
    std::cout << "----------------------------------------" <<std::endl;
    testRanks();
    return (EXIT_SUCCESS);
}

