/* 
 * File:   AgentTest.cpp
 * Author: volz
 *
 * Created on 20. Juni 2018, 16:03
 */

#include <stdlib.h>
#include <iostream>
#include "../Simulation/Agent.h"
#include "../Simulation/Card.h"
#include "../Simulation/Deck.h"
#include <stdexcept>


/*
 * Simple C++ Test Suite
 */

void testChoose(){
    std::cout << "Expected output: naive agent, always chooses max number" <<std::endl;
    int n = 10;
    int m = 4;
    int players = 2;
    Deck deck(n, m, 0, 10, 5);
    std::vector<std::vector<Card>> cards = deck.distribute(players);
    std::vector<int> playerLevel(4,0);
    std::vector<Agent>agent(players);
    for(int i=0; i<players; i++){
        std::vector<Card> hand = cards[i];
        agent[i] = Agent(playerLevel, deck);
        agent[i].pickUpCards(n/players, hand);
        for(int j=0; j<n/players; j++){
            std::cout << "chose " << agent[i].choose() << std::endl;
            Card card = agent[i].play();
            card.toString();
            card.printRanks();
        }
    }
    std::cout << "----------------------------------------" <<std::endl;
    
    Deck deck2(n, m, 0, 10, 5);
    std::vector<std::vector<Card>> cards2 = deck2.distribute(players);
    std::vector<int> playerLevel2(4,1);
    std::vector<Agent>agent2(players);
    for(int i=0; i<players; i++){
        std::vector<Card> hand = cards2[i];
        agent2[i] = Agent(playerLevel2, deck2);
        agent2[i].pickUpCards(n/players, hand);
        for(int j=0; j<n/players; j++){
            std::cout << "chose " << agent2[i].choose() << std::endl;
            Card card = agent2[i].play();
            card.toString();
            card.printRanks();
        }
    }
    
    std::cout << "Expected output: lvl 4 agent, always chooses highest rank" << std::endl;
}

void testPlay(){
    int n = 10;
    int m = 4;
    int players = 2;
    std::vector<double>values(n*m);
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    Deck deck(values, n, m);
    std::vector<std::vector<Card>> cards = deck.distribute(players);
    std::vector<int>playerLevel= {0,1,1,0};
    std::vector<Agent>agent(players);
    for(int i=0; i<players; i++){
        std::vector<Card> hand = cards[i];
        agent[i] = Agent(playerLevel, deck);
        agent[i].pickUpCards(n/players, hand);
    }
    for(int i =0; i<n/players; i++){
        Card card = agent[0].play();
        card.toString();
    }
    try{
       agent[0].play(); 
    }catch(const std::exception& e){
        std::cout << "Exception thrown" << std::endl;
    }
    
    std::cout << "Expected output: plays 5 cards (in order), then exception" <<std::endl;
    
}

void testPlayerLevel(){
    int n = 10;
    int m = 4;
    std::vector<double>values(n*m);
    for(int i=0; i<n*m; i++){
        values[i] = i;
    }
    Deck deck(values, n, m);
    std::vector<int>playerLevel1 = {0,1,1,0};
    Agent agent1(playerLevel1, deck);
    std::cout << "player level " << agent1.getLevel() << std::endl;
    std::cout <<"Expected output: player level 2"<< std::endl;
    std::vector<int>playerLevel2= {1,0,0,1};
    Agent agent2(playerLevel2, deck);
    std::cout << "player level " << agent2.getLevel() << std::endl;
    std::cout <<"Expected output: player level 2"<< std::endl;
}

void testRankRemains(){
    int n = 10;
    int m = 4;
    int players = 2;
    Deck deck(n, m, 0, 10, 5);
    std::vector<std::vector<Card>> cards = deck.distribute(players);
    std::vector<int>playerLevel = {0,1,1,0};
    std::vector<Agent>agent(players);
    for(int i=0; i<players; i++){
        std::vector<Card> hand = cards[i];
        agent[i] = Agent(playerLevel, deck);
        agent[i].pickUpCards(n/players, hand);
        agent[i].printRemainingCards();
    }
}

int main(int argc, char** argv) {
    testChoose();
    std::cout << "----------------------------------------" <<std::endl;
    testPlay();
    std::cout << "----------------------------------------" <<std::endl;
    testPlayerLevel();
    std::cout << "----------------------------------------" <<std::endl;
    testRankRemains();
    return (EXIT_SUCCESS);
}

