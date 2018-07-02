/* 
 * File:   GameTest.cpp
 * Author: volz
 *
 * Created on 20. Juni 2018, 17:31
 */

#include <stdlib.h>
#include <iostream>

#include "../Simulation/Game.h"
#include "../Simulation/Outcome.h"
/*
 * Simple C++ Test Suite
 */

void testSetup(){
    int n = 10;
    int m = 4;
    int players = 2;
    std::vector<double>values(n*m);
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    std::vector<double> min(m, 0);
    std::vector<double> max(m,100);
    Deck deck(values, n, m, min, max);
    std::vector<Agent>agents(players);
    std::vector<int>playerLevel1(4,0);
    agents[0] = Agent(playerLevel1, deck);
    std::vector<int>playerLevel2(4,1);
    agents[1] = Agent(playerLevel2, deck);


    Game game(deck, players, agents, 1);
    Outcome out(1);
    out = game.run(out,1);
    out.print();
    
    Deck deck2(values, n, m, min, max);
    Game game2(deck2, players, agents, 1);
    Outcome out2(1);
    out2 = game2.run(out2,1);
    out2.print();
    
    Deck deck3(values, n, m, min, max);
    Game game3(deck3, players, agents, 2);
    Outcome out3(1);
    out3 = game3.run(out3,1);
    out3.print();
}


void testRandomisation(){
    int n = 10;
    int m = 4;
    int players = 2;
    int rep = 100;
    std::vector<double>values(n*m);
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    std::vector<double> min(m, 0);
    std::vector<double> max(m,100);
    Deck deck(values, n, m, min, max);
    std::vector<Agent>agents(players);
    std::vector<int>playerLevel1(4,0);
    agents[0] = Agent(playerLevel1, deck);
    std::vector<int>playerLevel2(4,1);
    agents[1] = Agent(playerLevel2, deck);


    Game game(deck, players, agents, 1);
    Outcome out(rep);
    for(int i=0; i<rep; i++){
        out = game.run(out,0);
    }
    out.print();
    
    Deck deck2(values, n, m, min, max);
    Game game2(deck2, players, agents, 1);
    Outcome out2(rep);
    for(int i=0; i<rep; i++){
        out2 = game2.run(out2,0);
    }
    out2.print();
    
    Deck deck3(values, n, m, min, max);
    Game game3(deck3, players, agents, 100);
    Outcome out3(rep);
    for(int i=0; i<rep; i++){
        out3 = game3.run(out3,0);
    }
    out3.print();
}

void testRankUpdates(){
    int n = 10;
    int m = 4;
    int players=2;
    Deck deck(n, m, 0, 10, 5);
    std::vector<Card> cards = deck.getCards();
    std::vector<Agent>agents(players);
    std::vector<int>playerLevel1(4,0);
    agents[0] = Agent(playerLevel1, deck);
    std::vector<int>playerLevel2(4,1);
    agents[1] = Agent(playerLevel2, deck);
    
    Game game(deck, players, agents, 1);
    Outcome out(2);
    out = game.run(out,2);
    out = game.run(out,2);
    out.print();
}

int main(int argc, char** argv) {
    testSetup();
    std::cout << "----------------------------------------" <<std::endl;
    testRandomisation();
    std::cout << "----------------------------------------" <<std::endl;
    testRankUpdates();
    return (EXIT_SUCCESS);
}

