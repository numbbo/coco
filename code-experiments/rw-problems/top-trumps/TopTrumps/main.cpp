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

void printOutput(double* value, int obj){
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
    int inputDimension;
    double * values;
    bool first=true;
    int counter = 0;
    std::ifstream file("variables.txt");
    while (file >> readNumber){
        if(first){
            inputDimension= readNumber;
            values = new double[(int)inputDimension];
            first=false;
        }else{
            values[counter] = readNumber;
            counter++;
        }
    }
    
    int n = (int) inputDimension/m;
    
    if(obj>=5){
        d =2;
    }else{
        d=1;
    }
    
    double * result= new double[n];
    
    Deck deck(values, n, m);
    if(obj==0){
        result[0] = -deck.getHV();
    }else if(obj==1){
        result[0] = -deck.getSD();
    }else if(obj==5){
        result[0] = -deck.getHV();
        result[1] = -deck.getSD();
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
            result[0] = -out.getFairAgg();
        }else if(obj==3){
            result[0] = -out.getLeadChangeAgg();
        }else if(obj==4){
            result[0] = out.getTrickDiffAgg();
        }else if(obj==6){
            result[0] = -out.getFairAgg();
            result[1] = -out.getLeadChangeAgg();
        }else if(obj=7){
            result[0] = out.getTrickDiffAgg();
            result[1] = -out.getLeadChangeAgg();
        }
    }
    
    printOutput(result, d);
    return 0;
}



