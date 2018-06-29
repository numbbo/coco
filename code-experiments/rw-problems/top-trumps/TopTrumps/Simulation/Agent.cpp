#include "Agent.h"
#include <stdexcept>

Agent::Agent(){
    
}

Agent::Agent(int* playerLevel, Deck deck){
    this->currentCard = 0;
    this->deck = deck;
    this->playerLevel= playerLevel;
}

void Agent::pickUpCards(int handSize, Card* hand){
    this->handSize = handSize;
    this->hand = hand;
    this->currentCard=0;
    
}

void Agent::updateRanks(int * ranks){
    for(int k=0; k<this->deck.getM(); k++){
        for(int i=this->currentCard; i<this->handSize; i++){
            if(hand[i].getRank(k)>ranks[k]){
                hand[i].decreaseRank(k);
            }
        }   
    }
}


int Agent::choose(){
    //TODO should be depening on which is playerLevel, but doesn't matter for now
    int choice = 0;
    int best = 0;
    if(this->getLevel()==0){
        for(int i=0; i<this->deck.getM(); i++){
            int val = this->hand[this->currentCard].getValue(i);
            if(val>best){
                choice = i;
                best = val;
            }
        }
    }else{
        for(int i=0; i<this->deck.getM(); i++){
            int val = this->hand[this->currentCard].getRank(i);
            if(val>best){
                choice = i;
                best = val;
            }
        }  
    }
    return choice;
}

Card Agent::play(){
    if(this->currentCard>=this->handSize){
        throw std::invalid_argument("Out of cards");
    }
    this->currentCard++;
    return *(this->hand + (this->currentCard-1));
}

int Agent::getLevel(){
    int sum = 0;
    for(int i=0; i<this->deck.getM(); i++){
        sum+=this->playerLevel[i];
    }
    return sum;
}

void Agent::printRemainingCards(){
    for(int i=this->currentCard; i<this->handSize; i++){
        this->hand[i].toString();
        this->hand[i].printRanks();
    }
}