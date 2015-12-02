/*****************************************************************************

 avl.c - Source code for libavl

 Copyright (c) 1998  Michael H. Buselli <cosine@cosine.org>
 Copyright (c) 2000-2009  Wessel Dankers <wsl@fruit.je>

 This file is part of libavl.

 libavl is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as
 published by the Free Software Foundation, either version 3 of
 the License, or (at your option) any later version.

 libavl is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 You should have received a copy of the GNU General Public License
 and a copy of the GNU Lesser General Public License along with
 libavl.  If not, see <http://www.gnu.org/licenses/>.

 Augmented AVL-tree. Original by Michael H. Buselli <cosine@cosine.org>.

 Modified by Wessel Dankers <wsl@fruit.je> to add a bunch of bloat
 to the source code, change the interface and replace a few bugs.
 Mail him if you find any new bugs.

 Renamed and additionally modified by BOBBies to fit the COCO platform.

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/* We need either depths, counts or both (the latter being the default) */
#if !defined(AVL_DEPTH) && !defined(AVL_COUNT)
#define AVL_DEPTH
#define AVL_COUNT
#endif

/* User supplied function to compare two items like strcmp() does.
 * For example: compare(a,b) will return:
 *   -1  if a < b
 *    0  if a = b
 *    1  if a > b
 */
typedef int (*avl_compare_t)(const void *a, const void *b, void *userdata);

/* User supplied function to delete an item when a node is free()d.
 * If NULL, the item is not free()d.
 */
typedef void (*avl_free_t)(void *item, void *userdata);

#define AVL_CMP(a,b) ((a) < (b) ? -1 : (a) != (b))

#if defined(AVL_COUNT) && defined(AVL_DEPTH)
#define AVL_NODE_INITIALIZER(item) { 0, 0, 0, 0, 0, (item), 0, 0 }
#else
#define AVL_NODE_INITIALIZER(item) { 0, 0, 0, 0, 0, (item), 0 }
#endif

typedef struct avl_node {
  struct avl_node *prev;
  struct avl_node *next;
  struct avl_node *parent;
  struct avl_node *left;
  struct avl_node *right;
  void *item;
#ifdef AVL_COUNT
  unsigned long count;
#endif
#ifdef AVL_DEPTH
  unsigned char depth;
#endif
} avl_node_t;

#define AVL_TREE_INITIALIZER(cmp, free) { 0, 0, 0, (cmp), (free), {0}, 0, 0 }

typedef struct avl_tree {
  avl_node_t *top;
  avl_node_t *head;
  avl_node_t *tail;
  avl_compare_t cmpitem;
  avl_free_t freeitem;
  void *userdata;
  struct avl_allocator *allocator;
} avl_tree_t;

#define AVL_ALLOCATOR_INITIALIZER(alloc, dealloc) { (alloc), (dealloc) }

typedef avl_node_t *(*avl_allocate_t)(struct avl_allocator *);
typedef void (*avl_deallocate_t)(struct avl_allocator *, avl_node_t *);

typedef struct avl_allocator {
  avl_allocate_t allocate;
  avl_deallocate_t deallocate;
} avl_allocator_t;

static void avl_rebalance(avl_tree_t *, avl_node_t *);
static avl_node_t *avl_node_insert_after(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode);

#ifdef AVL_COUNT
#define NODE_COUNT(n)  ((n) ? (n)->count : 0)
#define L_COUNT(n)     (NODE_COUNT((n)->left))
#define R_COUNT(n)     (NODE_COUNT((n)->right))
#define CALC_COUNT(n)  (L_COUNT(n) + R_COUNT(n) + 1)
#endif

#ifdef AVL_DEPTH
#define NODE_DEPTH(n)  ((n) ? (n)->depth : 0)
#define L_DEPTH(n)     (NODE_DEPTH((n)->left))
#define R_DEPTH(n)     (NODE_DEPTH((n)->right))
#define CALC_DEPTH(n)  ((unsigned char)((L_DEPTH(n) > R_DEPTH(n) ? L_DEPTH(n) : R_DEPTH(n)) + 1))
#endif

const avl_node_t avl_node_0 = { 0 };
const avl_tree_t avl_tree_0 = { 0 };
const avl_allocator_t avl_allocator_0 = { 0 };

#define avl_const_node(x) ((avl_node_t *)(x))
#define avl_const_item(x) ((void *)(x))

static int avl_check_balance(avl_node_t *avlnode) {
#ifdef AVL_DEPTH
  int d;
  d = R_DEPTH(avlnode) - L_DEPTH(avlnode);
  return d < -1 ? -1 : d > 1;
#else
  /*  int d;
   *  d = ffs(R_COUNT(avl_node)) - ffs(L_COUNT(avl_node));
   *  d = d < -1 ? -1 : d > 1;
   */
#ifdef AVL_COUNT
  int pl, r;

  pl = ffs(L_COUNT(avlnode));
  r = R_COUNT(avlnode);

  if (r >> pl + 1)
  return 1;
  if (pl < 2 || r >> pl - 2)
  return 0;
  return -1;
#else
#error No balancing possible.
#endif
#endif
}

/* Commented to silence the compiler.

#ifdef AVL_COUNT
static unsigned long avl_count(const avl_tree_t *avltree) {
  if (!avltree)
    return 0;
  return NODE_COUNT(avltree->top);
}

static avl_node_t *avl_at(const avl_tree_t *avltree, unsigned long index) {
  avl_node_t *avlnode;
  unsigned long c;

  if (!avltree)
    return NULL;

  avlnode = avltree->top;

  while (avlnode) {
    c = L_COUNT(avlnode);

    if (index < c) {
      avlnode = avlnode->left;
    } else if (index > c) {
      avlnode = avlnode->right;
      index -= c + 1;
    } else {
      return avlnode;
    }
  }
  return NULL;
}

static unsigned long avl_index(const avl_node_t *avlnode) {
  avl_node_t *next;
  unsigned long c;

  if (!avlnode)
    return 0;

  c = L_COUNT(avlnode);

  while ((next = avlnode->parent)) {
    if (avlnode == next->right)
      c += L_COUNT(next) + 1;
    avlnode = next;
  }

  return c;
}
#endif

static const avl_node_t *avl_search_leftmost_equal(const avl_tree_t *tree, const avl_node_t *node,
    const void *item) {
  avl_compare_t cmp = tree->cmpitem;
  void *userdata = tree->userdata;
  const avl_node_t *r = node;

  for (;;) {
    for (;;) {
      node = node->left;
      if (!node)
        return r;
      if (cmp(item, node->item, userdata))
        break;
      r = node;
    }
    for (;;) {
      node = node->right;
      if (!node)
        return r;
      if (!cmp(item, node->item, userdata))
        break;
    }
    r = node;
  }

  return NULL; *//* To silence the compiler */
/*
}
*/

static const avl_node_t *avl_search_rightmost_equal(const avl_tree_t *tree,
                                                    const avl_node_t *node,
                                                    const void *item) {
  avl_compare_t cmp = tree->cmpitem;
  void *userdata = tree->userdata;
  const avl_node_t *r = node;

  for (;;) {
    for (;;) {
      node = node->right;
      if (!node)
        return r;
      if (cmp(item, node->item, userdata))
        break;
      r = node;
    }
    for (;;) {
      node = node->left;
      if (!node)
        return r;
      if (!cmp(item, node->item, userdata))
        break;
    }
    r = node;
  }

  return NULL; /* To silence the compiler */
}

/* Searches for an item, returning either some exact
 * match, or (if no exact match could be found) the first (leftmost)
 * of the nodes that have an item larger than the search item.
 * If exact is not NULL, *exact will be set to:
 *    0  if the returned node is unequal or NULL
 *    1  if the returned node is equal
 * Returns NULL if no equal or larger element could be found.
 * O(lg n) */

/* Commented to silence the compiler.
static avl_node_t *avl_search_leftish(const avl_tree_t *tree, const void *item, int *exact) {
  avl_node_t *node;
  avl_compare_t cmp;
  void *userdata;
  int c;

  if (!exact)
    exact = &c;

  if (!tree)
    return *exact = 0, (avl_node_t *) NULL;

  node = tree->top;
  if (!node)
    return *exact = 0, (avl_node_t *) NULL;

  cmp = tree->cmpitem;
  userdata = tree->userdata;

  for (;;) {
    c = cmp(item, node->item, userdata);

    if (c < 0) {
      if (node->left)
        node = node->left;
      else
        return *exact = 0, node;
    } else if (c > 0) {
      if (node->right)
        node = node->right;
      else
        return *exact = 0, node->next;
    } else {
      return *exact = 1, node;
    }
  }

  return NULL; *//* To silence the compiler */
/*
}
*/

/* Searches for an item, returning either some exact
 * match, or (if no exact match could be found) the last (rightmost)
 * of the nodes that have an item smaller than the search item.
 * If exact is not NULL, *exact will be set to:
 *    0  if the returned node is unequal or NULL
 *    1  if the returned node is equal
 * Returns NULL if no equal or smaller element could be found.
 * O(lg n) */
static avl_node_t *avl_search_rightish(const avl_tree_t *tree, const void *item, int *exact) {
  avl_node_t *node;
  avl_compare_t cmp;
  void *userdata;
  int c;

  if (!exact)
    exact = &c;

  if (!tree)
    return *exact = 0, (avl_node_t *) NULL;

  node = tree->top;
  if (!node)
    return *exact = 0, (avl_node_t *) NULL;

  cmp = tree->cmpitem;
  userdata = tree->userdata;

  for (;;) {
    c = cmp(item, node->item, userdata);

    if (c < 0) {
      if (node->left)
        node = node->left;
      else
        return *exact = 0, node->prev;
    } else if (c > 0) {
      if (node->right)
        node = node->right;
      else
        return *exact = 0, node;
    } else {
      return *exact = 1, node;
    }
  }

  return NULL; /* To silence the compiler */
}

/* Commented to silence the compiler.

static avl_node_t *avl_item_search_left(const avl_tree_t *tree, const void *item, int *exact) {
  avl_node_t *node;
  int c;

  if (!exact)
    exact = &c;

  if (!tree)
    return *exact = 0, (avl_node_t *) NULL;

  node = avl_search_leftish(tree, item, exact);
  if (*exact)
    return avl_const_node(avl_search_leftmost_equal(tree, node, item));

  return avl_const_node(node);
}
*/

/* Searches for an item, returning either the last (rightmost) exact
 * match, or (if no exact match could be found) the last (rightmost)
 * of the nodes that have an item smaller than the search item.
 * If exact is not NULL, *exact will be set to:
 *    0  if the returned node is inequal or NULL
 *    1  if the returned node is equal
 * Returns NULL if no equal or smaller element could be found.
 * O(lg n) */
static avl_node_t *avl_item_search_right(const avl_tree_t *tree, const void *item, int *exact) {
  const avl_node_t *node;
  int c;

  if (!exact)
    exact = &c;

  node = avl_search_rightish(tree, item, exact);
  if (*exact)
    return avl_const_node(avl_search_rightmost_equal(tree, node, item));

  return avl_const_node(node);
}

/* Searches for the item in the tree and returns a matching node if found
 * or NULL if not.
 * O(lg n) */
static avl_node_t *avl_item_search(const avl_tree_t *avltree, const void *item) {
  int c;
  avl_node_t *n;
  n = avl_search_rightish(avltree, item, &c);
  return c ? n : NULL;
}

/* Initializes a new tree for elements that will be ordered using
 * the supplied strcmp()-like function.
 * Returns the value of avltree (even if it's NULL).
 * O(1) */
static avl_tree_t *avl_tree_init(avl_tree_t *avltree, avl_compare_t cmp, avl_free_t free) {
  if (avltree) {
    avltree->head = NULL;
    avltree->tail = NULL;
    avltree->top = NULL;
    avltree->cmpitem = cmp;
    avltree->freeitem = free;
    avltree->userdata = NULL;
    avltree->allocator = NULL;
  }
  return avltree;
}

/* Allocates and initializes a new tree for elements that will be
 * ordered using the supplied strcmp()-like function.
 * Returns NULL if memory could not be allocated.
 * O(1) */
static avl_tree_t *avl_tree_construct(avl_compare_t cmp, avl_free_t free) {
  return avl_tree_init((avl_tree_t *) malloc(sizeof(avl_tree_t)), cmp, free);
}

/* Reinitializes the tree structure for reuse. Nothing is free()d.
 * Compare and free functions are left alone.
 * Returns the value of avltree (even if it's NULL).
 * O(1) */
static avl_tree_t *avl_tree_clear(avl_tree_t *avltree) {
  if (avltree)
    avltree->top = avltree->head = avltree->tail = NULL;
  return avltree;
}

static void avl_node_free(avl_tree_t *avltree, avl_node_t *node) {
  avl_allocator_t *allocator;
  avl_deallocate_t deallocate;

  allocator = avltree->allocator;
  if (allocator) {
    deallocate = allocator->deallocate;
    if (deallocate)
      deallocate(allocator, node);
  } else {
    free(node);
  }
}

/* Free()s all nodes in the tree but leaves the tree itself.
 * If the tree's free is not NULL it will be invoked on every item.
 * Returns the value of avltree (even if it's NULL).
 * O(n) */
static avl_tree_t *avl_tree_purge(avl_tree_t *avltree) {
  avl_node_t *node, *next;
  avl_free_t func;
  avl_allocator_t *allocator;
  avl_deallocate_t deallocate;
  void *userdata;

  if (!avltree)
    return NULL;

  userdata = avltree->userdata;

  func = avltree->freeitem;
  allocator = avltree->allocator;
  deallocate = allocator ? allocator->deallocate : (avl_deallocate_t) NULL;

  for (node = avltree->head; node; node = next) {
    next = node->next;
    if (func)
      func(node->item, userdata);
    if (allocator) {
      if (deallocate)
        deallocate(allocator, node);
    } else {
      free(node);
    }
  }

  return avl_tree_clear(avltree);
}

/* Frees the entire tree efficiently. Nodes will be free()d.
 * If the tree's free is not NULL it will be invoked on every item.
 * O(n) */
static void avl_tree_destruct(avl_tree_t *avltree) {
  if (!avltree)
    return;
  (void) avl_tree_purge(avltree);
  free(avltree);
}

static void avl_node_clear(avl_node_t *newnode) {
  newnode->left = newnode->right = NULL;
#   ifdef AVL_COUNT
  newnode->count = 1;
#   endif
#   ifdef AVL_DEPTH
  newnode->depth = 1;
#   endif
}

/* Initializes memory for use as a node.
 * Returns the value of avlnode (even if it's NULL).
 * O(1) */
static avl_node_t *avl_node_init(avl_node_t *newnode, const void *item) {
  if (newnode)
    newnode->item = avl_const_item(item);
  return newnode;
}

/* Allocates and initializes memory for use as a node.
 * Returns the value of avlnode (or NULL if the allocation failed).
 * O(1) */
static avl_node_t *avl_alloc(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;
  avl_allocator_t *allocator = avltree ? avltree->allocator : (avl_allocator_t *) NULL;
  avl_allocate_t allocate;
  if (allocator) {
    allocate = allocator->allocate;
    if (allocator) {
      newnode = allocate(allocator);
    } else {
      errno = ENOSYS;
      newnode = NULL;
    }
  } else {
    newnode = (avl_node_t *) malloc(sizeof *newnode);
  }
  return avl_node_init(newnode, item);
}

/* Insert a node in an empty tree. If avl_node is NULL, the tree will be
 * cleared and ready for re-use.
 * If the tree is not empty, the old nodes are left dangling.
 * O(1) */
static avl_node_t *avl_insert_top(avl_tree_t *avltree, avl_node_t *newnode) {
  avl_node_clear(newnode);
  newnode->prev = newnode->next = newnode->parent = NULL;
  avltree->head = avltree->tail = avltree->top = newnode;
  return newnode;
}

/* Insert a node before another node. Returns the new node.
 * If old is NULL, the item is appended to the tree.
 * O(lg n) */
static avl_node_t *avl_node_insert_before(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode) {
  if (!avltree || !newnode)
    return NULL;

  if (!node)
    return
        avltree->tail ?
            avl_node_insert_after(avltree, avltree->tail, newnode) : avl_insert_top(avltree, newnode);

  if (node->left)
    return avl_node_insert_after(avltree, node->prev, newnode);

  avl_node_clear(newnode);

  newnode->next = node;
  newnode->parent = node;

  newnode->prev = node->prev;
  if (node->prev)
    node->prev->next = newnode;
  else
    avltree->head = newnode;
  node->prev = newnode;

  node->left = newnode;
  avl_rebalance(avltree, node);
  return newnode;
}

/* Insert a node after another node. Returns the new node.
 * If old is NULL, the item is prepended to the tree.
 * O(lg n) */
static avl_node_t *avl_node_insert_after(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode) {
  if (!avltree || !newnode)
    return NULL;

  if (!node)
    return
        avltree->head ?
            avl_node_insert_before(avltree, avltree->head, newnode) : avl_insert_top(avltree, newnode);

  if (node->right)
    return avl_node_insert_before(avltree, node->next, newnode);

  avl_node_clear(newnode);

  newnode->prev = node;
  newnode->parent = node;

  newnode->next = node->next;
  if (node->next)
    node->next->prev = newnode;
  else
    avltree->tail = newnode;
  node->next = newnode;

  node->right = newnode;
  avl_rebalance(avltree, node);
  return newnode;
}

/* Insert a node into the tree and return it.
 * Returns NULL if an equal node is already in the tree.
 * O(lg n) */
static avl_node_t *avl_node_insert(avl_tree_t *avltree, avl_node_t *newnode) {
  avl_node_t *node;
  int c;

  node = avl_search_rightish(avltree, newnode->item, &c);
  return c ? NULL : avl_node_insert_after(avltree, node, newnode);
}

/* Commented to silence the compiler.

static avl_node_t *avl_node_insert_left(avl_tree_t *avltree, avl_node_t *newnode) {
  return avl_node_insert_before(avltree, avl_item_search_left(avltree, newnode->item, NULL), newnode);
}

static avl_node_t *avl_node_insert_right(avl_tree_t *avltree, avl_node_t *newnode) {
  return avl_node_insert_after(avltree, avl_item_search_right(avltree, newnode->item, NULL), newnode);
}

static avl_node_t *avl_node_insert_somewhere(avl_tree_t *avltree, avl_node_t *newnode) {
  return avl_node_insert_after(avltree, avl_search_rightish(avltree, newnode->item, NULL), newnode);
}
*/

/* Insert an item into the tree and return the new node.
 * Returns NULL and sets errno if memory for the new node could not be
 * allocated or if the node is already in the tree (EEXIST).
 * O(lg n) */
static avl_node_t *avl_item_insert(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode) {
    if (avl_node_insert(avltree, newnode))
      return newnode;
    avl_node_free(avltree, newnode);
    errno = EEXIST;
  }
  return NULL;
}

/* Commented to silence the compiler.

static avl_node_t *avl_item_insert_somewhere(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_somewhere(avltree, newnode);
  return NULL;
}

static avl_node_t *avl_item_insert_before(avl_tree_t *avltree, avl_node_t *node, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_before(avltree, node, newnode);
  return NULL;
}

static avl_node_t *avl_item_insert_after(avl_tree_t *avltree, avl_node_t *node, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_after(avltree, node, newnode);
  return NULL;
}

static avl_node_t *avl_item_insert_left(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_left(avltree, newnode);
  return NULL;
}

static avl_node_t *avl_item_insert_right(avl_tree_t *avltree, const void *item) {
  avl_node_t *newnode;

  if (!avltree)
    return errno = EFAULT, (avl_node_t *) NULL;

  newnode = avl_alloc(avltree, item);
  if (newnode)
    return avl_node_insert_right(avltree, newnode);
  return NULL;
}
*/

/* Deletes a node from the tree.
 * Returns the value of the node (even if it's NULL).
 * The item will NOT be free()d regardless of the tree's free handler.
 * This function comes in handy if you need to update the search key.
 * O(lg n) */
static avl_node_t *avl_node_unlink(avl_tree_t *avltree, avl_node_t *avlnode) {
  avl_node_t *parent;
  avl_node_t **superparent;
  avl_node_t *subst, *left, *right;
  avl_node_t *balnode;

  if (!avltree || !avlnode)
    return NULL;

  if (avlnode->prev)
    avlnode->prev->next = avlnode->next;
  else
    avltree->head = avlnode->next;

  if (avlnode->next)
    avlnode->next->prev = avlnode->prev;
  else
    avltree->tail = avlnode->prev;

  parent = avlnode->parent;

  superparent = parent ? avlnode == parent->left ? &parent->left : &parent->right : &avltree->top;

  left = avlnode->left;
  right = avlnode->right;
  if (!left) {
    *superparent = right;
    if (right)
      right->parent = parent;
    balnode = parent;
  } else if (!right) {
    *superparent = left;
    left->parent = parent;
    balnode = parent;
  } else {
    subst = avlnode->prev;
    if (subst == left) {
      balnode = subst;
    } else {
      balnode = subst->parent;
      balnode->right = subst->left;
      if (balnode->right)
        balnode->right->parent = balnode;
      subst->left = left;
      left->parent = subst;
    }
    subst->right = right;
    subst->parent = parent;
    right->parent = subst;
    *superparent = subst;
  }

  avl_rebalance(avltree, balnode);

  return avlnode;
}

/* Deletes a node from the tree. Returns immediately if the node is NULL.
 * If the tree's free is not NULL, it is invoked on the item.
 * If it is, returns the item. In all other cases returns NULL.
 * O(lg n) */
static void *avl_node_delete(avl_tree_t *avltree, avl_node_t *avlnode) {
  void *item = NULL;
  if (avlnode) {
    item = avlnode->item;
    (void) avl_node_unlink(avltree, avlnode);
    if (avltree->freeitem)
      avltree->freeitem(item, avltree->userdata);
    avl_node_free(avltree, avlnode);
  }
  return item;
}

/* Searches for an item in the tree and deletes it if found.
 * If the tree's free is not NULL, it is invoked on the item.
 * If it is, returns the item. In all other cases returns NULL.
 * O(lg n) */
static void *avl_item_delete(avl_tree_t *avltree, const void *item) {
  return avl_node_delete(avltree, avl_item_search(avltree, item));
}

/* Commented to silence the compiler.

static avl_node_t *avl_node_fixup(avl_tree_t *avltree, avl_node_t *newnode) {
  avl_node_t *oldnode = NULL, *node;

  if (!avltree || !newnode)
    return NULL;

  node = newnode->prev;
  if (node) {
    oldnode = node->next;
    node->next = newnode;
  } else {
    avltree->head = newnode;
  }

  node = newnode->next;
  if (node) {
    oldnode = node->prev;
    node->prev = newnode;
  } else {
    avltree->tail = newnode;
  }

  node = newnode->parent;
  if (node) {
    if (node->left == oldnode)
      node->left = newnode;
    else
      node->right = newnode;
  } else {
    oldnode = avltree->top;
    avltree->top = newnode;
  }

  return oldnode;
}
*/

/**
 * avl_rebalance:
 * Rebalances the tree if one side becomes too heavy.  This function
 * assumes that both subtrees are AVL-trees with consistent data.  The
 * function has the additional side effect of recalculating the count of
 * the tree at this node.  It should be noted that at the return of this
 * function, if a rebalance takes place, the top of this subtree is no
 * longer going to be the same node.
 */
static void avl_rebalance(avl_tree_t *avltree, avl_node_t *avlnode) {
  avl_node_t *child;
  avl_node_t *gchild;
  avl_node_t *parent;
  avl_node_t **superparent;

  parent = avlnode;

  while (avlnode) {
    parent = avlnode->parent;

    superparent = parent ? avlnode == parent->left ? &parent->left : &parent->right : &avltree->top;

    switch (avl_check_balance(avlnode)) {
    case -1:
      child = avlnode->left;
#           ifdef AVL_DEPTH
      if (L_DEPTH(child) >= R_DEPTH(child)) {
#           else
#           ifdef AVL_COUNT
        if (L_COUNT(child) >= R_COUNT(child)) {
#           else
#           error No balancing possible.
#           endif
#           endif
        avlnode->left = child->right;
        if (avlnode->left)
          avlnode->left->parent = avlnode;
        child->right = avlnode;
        avlnode->parent = child;
        *superparent = child;
        child->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = CALC_COUNT(avlnode);
        child->count = CALC_COUNT(child);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = CALC_DEPTH(avlnode);
        child->depth = CALC_DEPTH(child);
#               endif
      } else {
        gchild = child->right;
        avlnode->left = gchild->right;
        if (avlnode->left)
          avlnode->left->parent = avlnode;
        child->right = gchild->left;
        if (child->right)
          child->right->parent = child;
        gchild->right = avlnode;
        if (gchild->right)
          gchild->right->parent = gchild;
        gchild->left = child;
        if (gchild->left)
          gchild->left->parent = gchild;
        *superparent = gchild;
        gchild->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = CALC_COUNT(avlnode);
        child->count = CALC_COUNT(child);
        gchild->count = CALC_COUNT(gchild);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = CALC_DEPTH(avlnode);
        child->depth = CALC_DEPTH(child);
        gchild->depth = CALC_DEPTH(gchild);
#               endif
      }
      break;
    case 1:
      child = avlnode->right;
#           ifdef AVL_DEPTH
      if (R_DEPTH(child) >= L_DEPTH(child)) {
#           else
#           ifdef AVL_COUNT
        if (R_COUNT(child) >= L_COUNT(child)) {
#           else
#           error No balancing possible.
#           endif
#           endif
        avlnode->right = child->left;
        if (avlnode->right)
          avlnode->right->parent = avlnode;
        child->left = avlnode;
        avlnode->parent = child;
        *superparent = child;
        child->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = CALC_COUNT(avlnode);
        child->count = CALC_COUNT(child);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = CALC_DEPTH(avlnode);
        child->depth = CALC_DEPTH(child);
#               endif
      } else {
        gchild = child->left;
        avlnode->right = gchild->left;
        if (avlnode->right)
          avlnode->right->parent = avlnode;
        child->left = gchild->right;
        if (child->left)
          child->left->parent = child;
        gchild->left = avlnode;
        if (gchild->left)
          gchild->left->parent = gchild;
        gchild->right = child;
        if (gchild->right)
          gchild->right->parent = gchild;
        *superparent = gchild;
        gchild->parent = parent;
#               ifdef AVL_COUNT
        avlnode->count = CALC_COUNT(avlnode);
        child->count = CALC_COUNT(child);
        gchild->count = CALC_COUNT(gchild);
#               endif
#               ifdef AVL_DEPTH
        avlnode->depth = CALC_DEPTH(avlnode);
        child->depth = CALC_DEPTH(child);
        gchild->depth = CALC_DEPTH(gchild);
#               endif
      }
      break;
    default:
#           ifdef AVL_COUNT
      avlnode->count = CALC_COUNT(avlnode);
#           endif
#           ifdef AVL_DEPTH
      avlnode->depth = CALC_DEPTH(avlnode);
#           endif
    }
    avlnode = parent;
  }
}
