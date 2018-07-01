#include "Game.h"
#include <iostream>
Game::Game(Deck deck, int players, std::vector<Agent> agents, int seed){
    this->agents = agents;
    this->players = players;
    this->deck = deck;
    this->re = std::default_random_engine(seed);
    
    int maxLevel = 0;
    this->bestPlayer = 0;
    for(int i=0; i<this->players; i++){
        int lvl = this->agents[i].getLevel();
        if(lvl>maxLevel){
            maxLevel = lvl;
            this->bestPlayer = i;
        }
    }
}

Outcome Game::run(Outcome out, int verbose){
    this->deck.computeRanks();
    this->deck.shuffle();
    int total_rounds = this->deck.getN()/this->players;
    
    std::vector<std::vector<Card>> cards = deck.distribute(players);
    for(int i=0; i<players; i++){
        std::vector<Card> hand = cards[i];
        agents[i].pickUpCards(total_rounds, hand);
    }
    int won_last = (rand() % this->players);
    
    if(verbose>=1){
        for(int i=0; i<players; i++){
            std::cout << "Hand player " << i << std::endl;
            agents[i].printRemainingCards();
        }
        std::cout<< "Game starts with " << won_last << std::endl;
    }
    
    std::vector<int>tricks(this->players, 0);
    for(int i=0; i<total_rounds; i++){
        int winner = round(won_last, cards, verbose);
        tricks[winner]++;
        if(winner!=won_last){
            out.leadChanged();
            won_last = winner;
        }
    }
    int first = 0;
    int last = 0;
    int maxTricks = 0;
    int minTricks = total_rounds;
    for(int i=0; i<this->players; i++){
        if(maxTricks < tricks[i]){
            maxTricks = tricks[i];
            first = i;
        }
        if(minTricks > tricks[i]){
            minTricks = tricks[i];
            last = i;
        }
    }
    if(first == this->bestPlayer){
        out.betterPlayerWon();
    }
    out.addTrickDiff(maxTricks-minTricks);
    return out;
}

int Game::round(int won_last, std::vector<std::vector<Card>> cards, int verbose){
    int category = this->agents[won_last].choose();
    if(verbose>=1){
        std::cout << "Chose " << category << std::endl;
    }
    int winner = 0;
    double best = 0;
    for(int i=0; i<players; i++){
        Card played = this->agents[i].play();
        if(verbose>=1){
            played.toString();
        }
        for(int j=0; j<this->players; j++){
            this->agents[j].updateRanks(played.getRanks());
        }
        double value = played.getValue(category);
        if(best<value){
            best = value;
            winner = i;
        }else if(best==value){
            int rnd = rand() % 100;
            if(rnd>=50){
                best = value;
                winner = i;
            }
        }
    }
    if(verbose>=2){
        for(int i=0; i<this->players; i++){
            std::cout << "Player " <<i << " remaining cards after rank update" <<std::endl;
            this->agents[i].printRemainingCards();  
        }
    }else if(verbose>=1){
        std::cout << "Winner: " << winner << std::endl;
    }
    return winner; 
}
