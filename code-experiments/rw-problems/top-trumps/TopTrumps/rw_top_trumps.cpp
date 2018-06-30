/*
 * rw_top_trumps.cpp
 *
 *  Created on: 29. jun. 2018
 *      Author: Tea Tusar
 */
#include "rw_top_trumps.h"

#include "Simulation/Game.h"
#include <assert.h>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

void top_trumps_evaluate(size_t function, size_t instance, size_t size_x,
    double *x, size_t size_y, double *y) {

  int seed = (int) instance;
  int obj = (int) function;
  int rep = 100;
  int m = 4;
  int players = 2;
  int n = (int) size_x / m;

  assert(((obj <= 5) && (size_y == 1)) || ((obj >= 6) && (size_y == 2)));

  std::vector<double> y_vector(n);
  std::vector<double> x_vector(x, x + size_x);

  Deck deck(x_vector, n, m);
  if (obj == 1) {
    y_vector[0] = -deck.getHV();
  } else if (obj == 2) {
    y_vector[0] = -deck.getSD();
  } else if (obj == 6) {
    y_vector[0] = -deck.getHV();
    y_vector[1] = -deck.getSD();
  } else {
    std::vector<Agent> agents(players);
    std::vector<int> playerLevel1(4, 0);
    agents[0] = Agent(playerLevel1, deck);
    std::vector<int> playerLevel2(4, 1);
    agents[1] = Agent(playerLevel2, deck);

    Game game(deck, players, agents, seed);
    Outcome out(rep);
    for (int i = 0; i < rep; i++) {
      out = game.run(out, 0);
    }

    if (obj == 3) {
      y_vector[0] = -out.getFairAgg();
    } else if (obj == 4) {
      y_vector[0] = -out.getLeadChangeAgg();
    } else if (obj == 5) {
      y_vector[0] = out.getTrickDiffAgg();
    } else if (obj == 7) {
      y_vector[0] = -out.getFairAgg();
      y_vector[1] = -out.getLeadChangeAgg();
    } else if (obj == 8) {
      y_vector[0] = out.getTrickDiffAgg();
      y_vector[1] = -out.getLeadChangeAgg();
    }
  }

  for (size_t i = 0; i < size_y; i++)
    y[i] = y_vector[i];
}

void top_trumps_test(void) {
  std::cout << "Top trumps is working!\n";
}

#ifdef __cplusplus
}
#endif

