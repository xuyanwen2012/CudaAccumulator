#pragma once

#include <type_traits>

template <typename T,
          typename = std::enable_if<std::is_floating_point_v<T>, T>>
struct body
{
	T x;
	T y;
	T mass;
};
