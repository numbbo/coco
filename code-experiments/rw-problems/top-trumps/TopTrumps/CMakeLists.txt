cmake_minimum_required(VERSION 2.8)

project(TopTrumps)

# Executable
add_executable(TopTrumpsExec main.cpp Simulation/Card.cpp Simulation/Deck.cpp Simulation/Agent.cpp Simulation/Game.cpp Simulation/Outcome.cpp utils/hoy.cpp)
set_property(TARGET TopTrumpsExec PROPERTY CXX_STANDARD 11)

#Test Executables
add_executable(DeckTest tests/DeckTest.cpp Simulation/Card.cpp Simulation/Deck.cpp utils/hoy.cpp)
set_property(TARGET DeckTest PROPERTY CXX_STANDARD 11)

add_executable(AgentTest tests/AgentTest.cpp Simulation/Card.cpp Simulation/Deck.cpp Simulation/Agent.cpp utils/hoy.cpp)
set_property(TARGET AgentTest PROPERTY CXX_STANDARD 11)

add_executable(GameTest tests/GameTest.cpp Simulation/Card.cpp Simulation/Deck.cpp Simulation/Agent.cpp Simulation/Game.cpp Simulation/Outcome.cpp utils/hoy.cpp)
set_property(TARGET GameTest PROPERTY CXX_STANDARD 11)
