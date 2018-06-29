/*
 * top_trumps.cpp
 *
 *  Created on: 29. jun. 2018
 *      Author: Tea Tusar
 */
#include "Simulation/Game.h"
#include "top_trumps.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

double *top_trumps_evaluate(size_t function, size_t instance, size_t size_x,
    double *x, size_t size_y) {

    int seed = (int)instance;
    int obj = (int)function;
    int rep = 100;
    int m = 4;
    int players = 2;
    int n = (int) size_x/m;

    assert((obj < 5) && (size_y == 1));
    assert((obj >= 5) && (size_y == 2));
    
    double *result = new double[n];
    
    Deck deck(x, n, m);
    if (obj == 0) {
        result[0] = -deck.getHV();
    } else if (obj == 1) {
        result[0] = -deck.getSD();
    } else if (obj == 5) {
        result[0] = -deck.getHV();
        result[1] = -deck.getSD();
    } else {
        Agent * agents = new Agent[players];
        int playerLevel1[4] = { 0 };
        agents[0] = *(new Agent(playerLevel1, deck));
        int playerLevel2[4] = { 1 };
        agents[1] = *(new Agent(playerLevel2, deck));

        Game game(deck, players, agents, seed);
        Outcome out(rep);
        for (int i = 0; i < rep; i++) {
          out = game.run(out, 0);
        }

        if (obj == 2) {
            result[0] = -out.getFairAgg();
        } else if (obj == 3) {
            result[0] = -out.getLeadChangeAgg();
        } else if (obj == 4) {
            result[0] = out.getTrickDiffAgg();
        } else if (obj == 6) {
            result[0] = -out.getFairAgg();
            result[1] = -out.getLeadChangeAgg();
        } else if (obj == 7) {
            result[0] = out.getTrickDiffAgg();
            result[1] = -out.getLeadChangeAgg();
        }
    }
    
    return result;
}

#ifdef __cplusplus
}
#endif



