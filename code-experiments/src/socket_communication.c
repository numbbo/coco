#include <stdio.h>
#include <string.h>
#include "coco_platform.h"

#define REPLY_BUFFER 1024 /* TODO: What would be a safe value to use here? */

static void socket_communication_evaluate(const char* host_name, const unsigned short port,
    const char *message, const size_t number_of_objectives, double *y) {
#if WINSOCK
  WSADATA wsa;
  SOCKET sock;
  struct sockaddr_in serv_addr;
  char reply[REPLY_BUFFER];
  int reply_size;

  /* Initialize Winsock */
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
    coco_error("socket_communication_evaluate(): Winsock initialization failed: %d", WSAGetLastError());
  }

  /* Create a socket */
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
    coco_error("socket_communication_evaluate(): Could not create socket: %d", WSAGetLastError());
  }

  serv_addr.sin_addr.s_addr = inet_addr(host_name);
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  /* Connect to the evaluator */
  if (connect(sock, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0) {
    coco_error("socket_communication_evaluate(): Connection failed");
  }

  /* Send message */
  if (send(sock, message, (int)strlen(message), 0) < 0) {
    coco_error("socket_communication_evaluate(): Send failed");
  }
  coco_info("Sent message: %s", message);

  /* Receive the reply */
  if ((reply_size = recv(sock, reply, REPLY_BUFFER, 0)) == SOCKET_ERROR) {
    coco_error("socket_communication_evaluate(): Receive failed");
  }
  coco_info("Received response: %s (%ld)", reply, reply_size);

  y[0] = 12;
#else
  int sock;
  struct sockaddr_in serv_addr;
  char reply[REPLY_BUFFER];
  long reply_size;

  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    coco_error("socket_communication_evaluate(): Socket creation error");
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  /* Convert IPv4 and IPv6 addresses from text to binary form */
  if (inet_pton(AF_INET, host_name, &serv_addr.sin_addr) <= 0) {
    coco_error("socket_communication_evaluate(): Invalid address / Address not supported");
  }

  /* Connect to the evaluator */
  if (connect(sock, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0) {
    coco_error("socket_communication_evaluate(): Connection Failed");
  }

  /* Send message */
  if (send(sock, message, strlen(message), 0) < 0) {
    coco_error("socket_communication_evaluate(): Send failed");
  }
  coco_info("Sent message: %s", message);

  reply_size = read(sock, reply, REPLY_BUFFER);
  coco_info("Received response: %s (%ld)", reply, reply_size);

  y[0] = 12;
#endif
}


