/* 
 * File:   main.cpp
 * Author: volz
 *
 * Created on 20. Juni 2018, 12:18
 */

#include <cstdlib>
#include "Simulation/Game.h"
#include <fstream>
#include <iostream>

using namespace std;

void printOutput(double value){
    ofstream file;
    file.open("objectives.txt");
    file << "1\n" << value;
    file.close();
}


/*
 * Input: Seed, objectiveIndicator, rep, n, x
 */
int main(int argc, char** argv) {
    int seed = atof(argv[1]);
    int obj = atof(argv[2]);
    int rep = atof(argv[3]);
    
    int m = 4;
    int players = 2;
    
    int n = atof(argv[4]);
    double values[n*m];
    for(int i=0; i<n*m; i++){
        values[i] = atof(argv[5+i]);
    }
    
    Deck deck(values, n, m);
    if(obj==0){
        printOutput(deck.getHV());
    }else if(obj==1){
        printOutput(deck.getSD());
    }else{
        Agent * agents = new Agent[players];
        int playerLevel1[4]= {0};
        agents[0] = *(new Agent(playerLevel1, deck));
        int playerLevel2[4] = {1};
        agents[1] = *(new Agent(playerLevel2, deck));


        Game game(deck, players, agents, seed);
        Outcome out(rep);
        for(int i=0; i<rep; i++){
            out = game.run(out,0);
        }
        
        if(obj==2){
            printOutput(out.getFairAgg());
        }else if(obj==3){
            printOutput(out.getLeadChangeAgg());
        }else if(obj==4){
            printOutput(out.getTrickDiffAgg());
        }
    }
    return 0;
}



