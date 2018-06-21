#include <vector>
#include <algorithm>
// Act like matlab's [Y,I] = SORT(X)
// Input:
//   unsorted  unsorted vector
// Output:
//   sorted     sorted vector, allowed to be same as unsorted
//   index_map  an index map such that sorted[i] = unsorted[index_map[i]]
template <class T>
void sort(
    std::vector<T> &unsorted,
    std::vector<T> &sorted,
    std::vector<size_t> &index_map);

// Act like matlab's Y = X[I]
// where I contains a vector of indices so that after,
// Y[j] = X[I[j]] for index j
// this implies that Y.size() == I.size()
// X and Y are allowed to be the same reference
template< class T >
void reorder(
  std::vector<T> & unordered, 
  std::vector<size_t> const & index_map, 
  std::vector<T> & ordered);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////


// Comparison struct used by sort
// http://bytes.com/topic/c/answers/132045-sort-get-index
template<class T> struct index_cmp 
{
  index_cmp(const T arr) : arr(arr) {}
  bool operator()(const size_t a, const size_t b) const
  { 
    return arr[a] < arr[b];
  }
  const T arr;
};

template <class T>
void sort(
  std::vector<T> & unsorted,
  std::vector<T> & sorted,
  std::vector<size_t> & index_map)
{
  // Original unsorted index map
  index_map.resize(unsorted.size());
  for(size_t i=0;i<unsorted.size();i++)
  {
    index_map[i] = i;
  }
  // Sort the index map, using unsorted for comparison
  sort(
    index_map.begin(), 
    index_map.end(), 
    index_cmp<std::vector<T>& >(unsorted));

  sorted.resize(unsorted.size());
  reorder(unsorted,index_map,sorted);
}

// This implementation is O(n), but also uses O(n) extra memory
template< class T >
void reorder(
  std::vector<T> & unordered, 
  std::vector<size_t> const & index_map, 
  std::vector<T> & ordered)
{
  // copy for the reorder according to index_map, because unsorted may also be
  // sorted
  std::vector<T> copy = unordered;
  ordered.resize(index_map.size());
  for(int i = 0; i<index_map.size();i++)
  {
    ordered[i] = copy[index_map[i]];
  }
}