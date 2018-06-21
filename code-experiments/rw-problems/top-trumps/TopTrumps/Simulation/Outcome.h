/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Outcome.h
 * Author: volz
 *
 * Created on 20. Juni 2018, 12:22
 */

#ifndef OUTCOME_H
#define OUTCOME_H

class Outcome{
public:
    Outcome(int games);
    
    double getLeadChangeAgg();
    double getTrickDiffAgg();
    double getFairAgg();
    
    void leadChanged();
    void betterPlayerWon();
    void addTrickDiff(int diff);
    
    void print();
private:
    int leadChange;
    int trickDiff;
    int fair;
    int games;
};

#endif /* OUTCOME_H */

