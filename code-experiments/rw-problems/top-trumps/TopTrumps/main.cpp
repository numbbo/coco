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

void printOutput(std::vector<double> value, int obj){
    ofstream file;
    file.open("objectives.txt");
    file << obj;
    for(int i=0; i<obj; i++){
        file << "\n" << value[i];
    }
    file.close();
}


/*
 * Input: Seed, objectiveIndicator, rep
 */
int main(int argc, char** argv) {
    int seed = atof(argv[1]);
    int obj = atof(argv[2]);
    int rep = atof(argv[3]);
    int d = 0;
    
    
    int m = 4;
    int players = 2;

    double readNumber;
    std::ifstream file("variables.txt");
    file >> readNumber;
    int inputDimension = (int)readNumber;
    std::vector<double>values(inputDimension);
    

    for(int i=0; i<inputDimension; i++){
        file >> readNumber;
        values[i] = readNumber;
    }
    file.close();
    
    int n = (int) inputDimension/m;
    if(obj>=6){
        d =2;
    }else{
        d=1;
    }
    
    std::vector<double>result(n);
    
    Deck deck(values, n, m);
    if(obj==1){
        result[0] = -deck.getHV();
    }else if(obj==2){
        result[0] = -deck.getSD();
    }else if(obj==6){
        result[0] = -deck.getHV();
        result[1] = -deck.getSD();
    }else{
        std::vector<Agent>agents(players);
        std::vector<int>playerLevel1(4,0);
        agents[0] = Agent(playerLevel1, deck);
        std::vector<int>playerLevel2(4,1);
        agents[1] = Agent(playerLevel2, deck);


        Game game(deck, players, agents, seed);
        Outcome out(rep);
        for(int i=0; i<rep; i++){
            out = game.run(out,0);
        }
        
        if(obj==3){
            result[0] = -out.getFairAgg();
        }else if(obj==4){
            result[0] = -out.getLeadChangeAgg();
        }else if(obj==5){
            result[0] = out.getTrickDiffAgg();
        }else if(obj==7){
            result[0] = -out.getFairAgg();
            result[1] = -out.getLeadChangeAgg();
        }else if(obj==8){
            result[0] = out.getTrickDiffAgg();
            result[1] = -out.getLeadChangeAgg();
        }
    }
    
    printOutput(result, d);
    return 0;
}



