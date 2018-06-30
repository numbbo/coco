/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Card.h
 * Author: volz
 *
 * Created on 20. Juni 2018, 12:27
 */

#ifndef CARD_H
#define CARD_H
#include <vector>

class Card{
public:
    Card();
    Card(std::vector<double> values, int m, int offset);
    
    std::vector<double> getValues();
    double getValue(int i);
    
    //void updateRanks(int * ranks);
    void decreaseRank(int i);
    void setRank(int rank, int j);
    std::vector<int> getRanks();
    int getRank(int i);
    
    void toString();
    void printRanks();
        
private:
    std::vector<double> values;
    int m;
    std::vector<int> ranks;
};


#endif /* CARD_H */

