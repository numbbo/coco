#include <algorithm>
#include <stdexcept>

#include "Deck.h"
#include "../utils/hoy.h"
#include "../utils/sort.h"

Deck::Deck(){
    
}

Deck::Deck(double* values, int n, int m){
    this->n = n;
    this->m = m;
    this->cards = new Card[n];
    for(int i=0; i<n*m; i+=m){
        cards[i/m] = *(new Card(&values[i], m));
    }
    this->computeRanks();
}

Deck::Deck(int n, int m, double min, double max, int seed){
    std::default_random_engine re(seed);
    this->n = n;
    this->m = m;
    this->cards = new Card[n];
    std::uniform_real_distribution<double> unif(min, max);
    for(int i=0; i<n; i++){
        double* card_values = new double[m];
        for(int j=0; j<m; j++){
            card_values[j] = unif(re);
        }
        cards[i] = *(new Card(card_values, m));
    }
    this->computeRanks();
}

void Deck::computeRanks(){
    for(int i=0; i<this->m; i++){
        std::vector<double> colValues;
        std::vector<size_t> idx;
        std::vector<double> b;
        colValues.resize(this->n);
        for(int j=0; j<this->n; j++){
            colValues[j] = this->cards[j].getValue(i);
        }
        sort(colValues, b, idx);
        for(int j=0; j<this->n; j++){
            this->cards[(int)idx[j]].setRank(j, i);
        }
    }

}


void Deck::shuffle(){
    std::random_shuffle(&cards[0], &cards[n]);
}

Card ** Deck::distribute(int players){
    if(n%players !=0){
        throw std::invalid_argument("deck can not be divided to players");
    }
    int hand = n/players;
    Card **distribution = new Card*[players];
    int counter = 0;
    for(int i=0; i<players; i++){
        distribution[i] = new Card[hand];
        for(int j=0; j<hand; j++){
            distribution[i][j] = cards[counter];
            counter++;
        }
    }
    return distribution;
}

Card * Deck::getCards(){
    return this->cards;
}

int Deck::getM(){
    return this->m;
}

int Deck::getN(){
    return this->n;
}


double Deck::getHV(){
    double refPoint[this->m] = {0};
    double * values = new double[this->n*this->m];
    int counter =0;
    for(int i =0; i<this->n; i++){
        Card card = this->cards[i];
        for(int j=0; j<this->m; j++){
            values[counter] = card.getValue(j);
            if(refPoint[j]<values[counter]){
                refPoint[j]= values[counter];
            }
            counter++;
        }
    }
    for(int i=0; i<this->m; i++){
        refPoint[i]+=1;
    }
    HVCalculator hv;
    //int dimension, int dataNumber, double* points, double* refPoint
    return hv.computeHV(this->m, this->n, values, refPoint);
}

double Deck::getSD(){
    double colSums[this->m] = {0};
    double mean=0;
    for(int i=0; i<this->n; i++){
        Card card = this->cards[i];
        for(int j=0; j<this->m; j++){
            colSums[j] += card.getValue(j)/this->n;
            mean +=card.getValue(j)/this->n;
        }
    }
    mean/=this->m;
    double sd = 0;
    for(int i=0; i<this->m; i++){
        sd+=(colSums[i] - mean)*(colSums[i] -mean);
    }
    sd/=(this->m-1);
    sd = std::sqrt(sd);
    return(sd);
}
