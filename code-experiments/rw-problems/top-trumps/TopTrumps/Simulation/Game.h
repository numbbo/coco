/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Game.h
 * Author: volz
 *
 * Created on 20. Juni 2018, 13:41
 */

#ifndef GAME_H
#define GAME_H

#include "Deck.h"
#include "Agent.h"
#include "Outcome.h"


class Game{
public:
    Game(Deck deck, int players, Agent * agents, int seed);
        
    Outcome run(Outcome out, int verbose=0);
protected:
    int round(int won_last, Card ** cards, int verbose=0);
    
private:
    Agent * agents;
    int players;
    Deck deck;
    std::default_random_engine re;
    
    int bestPlayer;//TODO Expects there to be exactly one best player
    
};


#endif /* GAME_H */

