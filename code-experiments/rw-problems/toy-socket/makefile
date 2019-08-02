CCFLAGS = -g -ggdb -std=c89 -pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion

all: socket_server

socket_server: socket_server.c
	$(CC) ${CCFLAGS} socket_server.c -o socket_server

clean:
	rm -f socket_server
