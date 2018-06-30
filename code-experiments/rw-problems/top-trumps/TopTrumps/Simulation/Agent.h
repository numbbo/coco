/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Agent.h
 * Author: volz
 *
 * Created on 20. Juni 2018, 12:21
 */

#ifndef AGENT_H
#define AGENT_H

#include "Card.h"
#include "Deck.h"

class Agent{
public:
    Agent();
    Agent(std::vector<int> playerLevel, Deck deck);
    
    void pickUpCards(int handSize, std::vector<Card> hand);
    void updateRanks(std::vector<int> ranks);
    int choose();
    Card play();
    
    int getLevel();
    void printRemainingCards();
    
private:
    std::vector<int> playerLevel;
    Deck deck;
    int handSize;
    
    std::vector<Card> hand;
    int currentCard;
};

#endif /* AGENT_H */

