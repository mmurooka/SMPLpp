/* Author: Masaki Murooka */

#ifndef GRID_UTILS_H
#define GRID_UTILS_H

#include <Eigen/Dense>

// Necessary for std::unordered_map
namespace std
{
template<>
struct hash<Eigen::Vector3i>
{
  size_t operator()(const Eigen::Vector3i & v) const
  {
    size_t seed = 0;
    for(int i = 0; i < 3; i++)
    {
      seed ^= v[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
} // namespace std

namespace smplpp
{
//! Size of each cell in grid [m]
constexpr double GRID_SCALE = 0.025;

/** \brief Convert grid index to grid position. */
template<typename Scalar = float>
Eigen::Matrix<Scalar, 3, 1> getGridPos(const Eigen::Vector3i & gridIdx)
{
  return GRID_SCALE * gridIdx.cast<Scalar>();
}

/** \brief Convert grid position to grid index. */
template<typename Scalar = float>
Eigen::Vector3i getGridIdx(const Eigen::Matrix<Scalar, 3, 1> & gridPos)
{
  return (gridPos / GRID_SCALE).array().round().matrix().template cast<int>();
}

/** \brief Convert grid position to grid index.

    Gets the lower index using floor instead of round.
*/
template<typename Scalar = float>
Eigen::Vector3i getGridIdxFloor(const Eigen::Matrix<Scalar, 3, 1> & gridPos)
{
  return (gridPos / GRID_SCALE).array().floor().matrix().template cast<int>();
}

/** \brief Convert grid position to grid index.

    Gets the upper index using ceil instead of round.
*/
template<typename Scalar = float>
Eigen::Vector3i getGridIdxCeil(const Eigen::Matrix<Scalar, 3, 1> & gridPos)
{
  return (gridPos / GRID_SCALE).array().ceil().matrix().template cast<int>();
}
} // namespace smplpp

#endif // GRID_UTILS_H
